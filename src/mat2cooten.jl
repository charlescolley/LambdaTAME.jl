include("/Users/ccolley/Code/GLANCE/TuranShadow.jl")   #local
#include("/homes/ccolley/Documents/Software/GLANCE/TuranShadow.jl")    #server
#using NumbersFromText
using FileIO: save
using JLD: save,load
using Random
using CSV
using MatrixNetworks
using Distributed

using PyCall
@pyimport pickle

#LVGNA_data = "/Users/ccolley/PycharmProjects/LambdaTAME/data/sparse_matrices/LVGNA/"  #local
LVGNA_data = "/homes/ccolley/Documents/Research/heresWaldo/data/MultiMagna_TAME"       #server
TENSOR_PATH = "/homes/ccolley/Documents/Research/TensorConstruction/tensors/"   #server




function write_tensor(edge_set,outputfile)

    k = length(edge_set[1])
    indices = zeros(k,length(edge_set))
    idx = 1
    n = -1
    for clique in edge_set
   
        indices[:,idx] = clique
        n_c = maximum(clique)
        if n < n_c
            n = n_c
        end
        idx += 1;
    end

   write_ssten(round.(Int,indices),n,outputfile)

end

function write_ssten(indices::Array{Int,2},n, filepath::String)
    file = open(filepath, "w")
    order = size(indices,1)

    header = "$(order) $(n) $(size(indices,2))\n"
    write(file,header);

    for (edge) in eachcol(indices)
	edge_string=""
        for v_i in edge
	     edge_string *= string(v_i," ")
        end
 	edge_string *= "\n"
	write(file,edge_string)
    end

    close(file)
end

#routine used for reading in edge files stored as .csv
function parse_csv(file)
     
     f = CSV.File(file,header=false, types=[Int,Int])
     B = zeros(length(f)-1,2) #ignore old header 
     B[:,1] = f.Column1[2:end]
     B[:,2] = f.Column2[2:end]
     B .+= 1

     A = sparse(B[:,1],B[:,2],1,maximum(B),maximum(B)) 
     A = max.(A,A')

     return A
end

#TODO: move to Experiments.jl?
function sample_motifs(matrix_file,output_path,orders,sample_counts)

    #check file format

    A = nothing
    
    file_ext = split(matrix_file,".")[end]
    if (file_ext == "smat")
        A = MatrixNetworks.readSMAT(matrix_file)
    elseif (file_ext == "csv")
    	A = parse_csv(matrix_file)
    end
    
    A = max.(A,A')

    seed = 4
    Random.seed!(seed)
    root_name = split(split(matrix_file,"/")[end],".")[1]

    for k in orders

    	#create a subfolder for given order
	order_folder = "order:$(k)/"
	local_folder = output_path*order_folder
	    #Note: assuming that output_path ends in '/'
		  
	if !isdir(output_path*order_folder)
	   mkdir(output_path*order_folder)
	end

        log_file = root_name*"-order:$(k)-seed:$(seed)-log.jld"

        runtimes = []
        unique_cliques = Set()

        prev_sample_count = 0
	    prev_motif_count = 0

        for samples in sort(sample_counts)
     	    #  -- find unique cliques  --  #
    	    ((_,cliques),t) = @timed TuranShadow(A,k,(samples-prev_sample_count))
	        prev_sample_count = samples

	    
            cliques = [sort(clique) for clique in cliques]
            sort!(cliques) #lexicographic ordered helps eliminate repeats
	
	    for clique in cliques
 	        push!(unique_cliques,clique)
	    end

	    #  terminate sample search if no new motifs found this iteration

	    motif_count = length(unique_cliques)
	    if (prev_motif_count == motif_count)
	       println("no new cliques found from last iter. motif_count=$(motif_count)")
	       break
	    else
	        prev_motif_count = motif_count
	    end
	    
	

	    tensor_name = split(root_name,".")[1]*"-order:$(k)-sample:$(samples)-seed:$(seed).ssten"
	        #NOTE: assuming matrix_file is of the form rootName.smat

   	    write_tensor(collect(unique_cliques),output_path*order_folder*tensor_name)
	
	    #  --  update log  --  #
	    push!(runtimes,(samples,t))
	    save(output_path*order_folder*log_file,"runtimes",runtimes)
	end
    end



end

function tensors_from_graph(A, orders::Array{Int,1}, sample_size::Int)


    tensors = Array{SymTensorUnweighted,1}(undef,length(orders))

    for order in orders
        edges, n = tensor_from_graph(A, order, sample_size)
    end

end

function tensors_from_graph(A, orders::Array{Int,1}, sample_sizes::Array{Int,1})

    @assert length(orders) == length(sample_sizes)

    for (order,sample) in zip(orders,sample_sizes)
        edges, n = tensor_from_graph(A, order, sample)
    end

end

function tensor_from_graph(A, order, t)

    _, cliques::Array{Array{Int64,1},1} = TuranShadow(A,order,t)

    reduce_to_unique_motifs!(cliques)

    indices = zeros(order,length(cliques))
    idx = 1
    n = -1
    
    for clique in cliques #is there a better way to do this? 
        indices[:,idx] = clique
        n_c = maximum(clique)
        if n < n_c
            n = n_c
        end
        idx += 1;
    end

    
    return SymTensorUnweighted(n,order,round.(Int,indices))

end

#TODO: 
function reduce_to_unique_motifs!(cliques::Array{Array{T,1},1}) where {T <: Int}

    order = length(cliques[1])

    for i = 1:length(cliques)
        sort!(cliques[i])
    end

    sort!(cliques)
    
    ix_drop = Array{Int,1}(undef,0)

    current_clique_idx = 1
    for i =2:length(cliques)
        
        if cliques[i] == cliques[current_clique_idx] #found a repeated clique, mark for removal
            push!(ix_drop,i)
        else #found a new clique, update ptr
            current_clique_idx = i 
        end

    end

    deleteat!(cliques,ix_drop)
    #return cliques, ix_drop
    #remove all repeated cliques
    #cliques = cliques[setdiff(begin:end, ix_drop)]

    
end

#=  -- Functions to remove?

function gather_runtime_data(tensor_folders,output_path =nothing)

    all_data_by_order = [parse_tensor_folder_runtimes(tensor_folder) for tensor_folder in tensor_folders]
    
    #find full range of orders and samples

    #orders = union([union(collect(keys(data))) for data in all_data_by_order])   
    orders = Set(vcat([collect(keys(data)) for data in all_data_by_order]...)) 
    #samples = union([union([Set(samples) for (samples,rt) in values(data)]) for data in all_data_by_order]...)
    samples = Set(vcat([vcat([s for (s,t) in values(data)]...) for data in all_data_by_order]...))
    
    order_idx  = Dict([(o,i) for (i,o) in enumerate(sort(collect(orders)))])
    sample_idx = Dict([(s,i) for (i,s) in enumerate(sort(collect(samples)))])

    runtimes = -ones(length(tensor_folders),length(order_idx),length(sample_idx))

    for (i,tensor_data) in enumerate(all_data_by_order)
        for (order,data) in tensor_data
            j = order_idx[order]

            for (s,rt) in zip(data...)
                runtimes[i,j,sample_idx[s]] = rt
            end
        end
    end


    if output_path !== nothing
        out = open(output_path,"w")
        saved_data = [runtimes,tensor_folders, [k for (k,v) in sort(order_idx)], [k for (k,v) in sort(sample_idx)]]
        pickle.dump(saved_data, out)
        close(out)
    end

    return runtimes, [k for (k,v) in sort(order_idx)], [k for (k,v) in sort(sample_idx)]
end

function parse_tensor_folder_runtimes(tensor_folder)
    # assuming tensor_folder ends in '/'

    data_by_order = Dict()


    for order_dir in readdir(tensor_folder)

        order = parse(Int,split(order_dir,":")[end])
        samples = []
        runtimes = []

        for file in readdir(tensor_folder * order_dir)

            if file[end-2:end] == "jld"
                data = load("$tensor_folder/$order_dir/$file")
                
                for (sample, runtime) in sort(data["runtimes"],by=x->x[1])
                    push!(samples,sample)
                    push!(runtimes,runtime)
                end

                data_by_order[order] = (samples,runtimes)
            end
        end

    end
    return data_by_order
end
=#