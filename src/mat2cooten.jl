#include("/Users/ccolley/Code/GLANCE/TuranShadow.jl")   #local
include("/homes/ccolley/Documents/Software/GLANCE/TuranShadow.jl")    #server
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


#=
    Small routine for finding motifs
=#
function sample_for_cliques()



    orders = [3,4,5,6,7,8,9,10]   
    sample_counts = [10000,50000,100000,1000000,10000000,100000000,1000000000] 

    #sever path
    synth_graph_loc = "/u/subspace_s3/ccolley/Documents/Research/graphAlignment/synthetic_alignments/synthGraphs/"

    inputs_outputs =  [
        ("duplicationPerturbedER-noiseModel:duplication-perturbation_p:0.75.smat","duplicationPerturbedER-noiseModel:duplication-perturbation_p:0.75/"), 
        ("duplicationPerturbedER-noiseModel:ER-premove:0.05.smat","duplicationPerturbedER-noiseModel:ER-premove:0.05/"),
        ("perturbed_human_PHY1-noiseModel:duplication-perturbation_p:0.75.smat","perturbed_human_PHY1-noiseModel:duplication-perturbation_p:0.75/"),
        ("perturbed_human_PHY1-noiseModel:ER-premove:0.05.smat","perturbed_human_PHY1-noiseModel:ER-premove:0.05/"),
    ]

    if nprocs() < (length(inputs_outputs) + 1)
        addprocs((length(inputs_outputs) + 1) - nprocs())
    end

    @everywhere include_string(Main,$(read("mat2cooten.jl",String)),"mat2cooten.jl")

    futures = []
    for (input,output) in inputs_outputs

        if !isdir(TENSOR_PATH*output)
            mkdir(TENSOR_PATH*output)
        end

        future = @spawn sample_motifs(synth_graph_loc*input,TENSOR_PATH*output,orders,sample_counts)  
        push!(futures,(future,input))

    end

    for (future, input) in futures
        fetch(future)
        println("joined $input")
    end

end

#TODO: this code is redundant, remove.
function parallel_parse(matrix_fldr,tensor_fldr)
    """
        Assuming that [matrix,tensor]_fldr ends in '/'
    """

    error("test this first")
    #find available matrices 
    matrix_files = readdir(matrix_fldr) 
    #filter for smat at the moment


    #orders = [3,4,5,6,7,8,9,10]
    #samples = [10000,50000,100000,1000000,10000000,100000000]

    orders = [3]
    samples = [100]

    futures = []

    for matrix in matrix_files 
        root_name = split(split(matrix_file,"/")[end],".")[1]
        
        # make a folder for each matrix to store built tensors
        if !isdir(tensor_fldr*root_name)
            mkdir(tensor_fldr*root_name)
        end

        future = @spawn sample_motifs(matrix_file,tensor_fldr*root_name,orders,sample_counts)

        push!(futures,(future))


    end

    #join the processes 
    for future in futures
        fetch(future)
    end

end


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


function snap_datasets()
    #src: huda's TuranShadow repo
    #;wget https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz

    M = readmatrix(Int,"as-skitter.txt")
    M .+= 1
    A = sparse(M[:,1],M[:,2],1,maximum(M),maximum(M)) 

end

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


function tensor_from_graph(A, k, t)

    _, cliques = TuranShadow(A,k,t)

    cliques = [sort(clique) for clique in cliques]
    sort!(cliques) #lexicographic ordered helps eliminate repeats
    cliques = Set(cliques) # ensure edges are unique

    indices = zeros(k,length(cliques))
    idx = 1
    n = -1
    for clique in cliques
        indices[:,idx] = clique
        n_c = maximum(clique)
        if n < n_c
            n = n_c
        end
        idx += 1;
    end

    return round.(Int,indices)

end

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