#=------------------------------------------------------------------------------
                        Formatting Routines
------------------------------------------------------------------------------=#
function produce_ssten_from_triangles(file)

    A = MatrixNetworks.readSMAT(file)
    (n,m) = size(A)
    if(n != m)
        println("rectangular")
    end

    T = collect(MatrixNetworks.triangles(A))

    output_file = alterfilename(file,".ssten",".",false)

    open(output_file,"w") do f
        write(f,"$(3)\t$(n)\t$(length(T))\n")

        for (v_i,v_j,v_k) in T
            write(f,"$(v_i)\t$(v_j)\t$(v_k)\t1.0\n")
        end
    end

end

function load_ThirdOrderSymTensor(filepath;enforceFormatting = true)

	#check path validity
	@assert filepath[end-5:end] == ".ssten"

    open(filepath) do file
		#preallocate from the header line
		order, n, m =
			[parse(Int,elem) for elem in split(chomp(readline(file)),'\t')]
        @assert order == 3

		indices = Array{Int,2}(undef,m,order)
		values = Array{Float64,1}(undef,m)

		i = 1
		@inbounds for line in eachline(file)
			entries = split(chomp(line),'\t')
			indices[i,:] = [parse(Int,elem) for elem in entries[1:end-1]]
			if enforceFormatting
				sort!(indices[i,:])
			end
			values[i] = parse(Float64,entries[end])
			i += 1
		end

		#check for 0 indexing
		zero_indexed = false

		@inbounds for i in 1:m
		    if indices[i,1] == 0
    			zero_indexed = true
				break
			end
	    end

		if zero_indexed
			indices .+= 1
		end

        return ThirdOrderSymTensor(n,indices,values)
    end
end

#=------------------------------------------------------------------------------
                        Local File Experiments Routines
------------------------------------------------------------------------------=#

#TODO: update to use kwargs properly
function self_alignment(ssten_files::Array{String,1};method="LambdaTAME")

    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files))

    for f in ssten_files
        A = load(f;type="COOTen")

        #permute tensor to align against
        p = sort(1:A.cubical_dimension,by=i->rand())
        A_permuted = COOTen(A.indices,A.vals,A.cubical_dimension)
        permute_tensor!(A_permuted,p)

        best_score, _, _ = align_tensors(A,A_permuted;method=method)

        println("$f aligned with $best_score")
    end


end

function align_tensors(graph_A_file::String,graph_B_file::String;
                       ThirdOrderSparse=true,kwargs...)

    if ThirdOrderSparse
        A = load_ThirdOrderSymTensor(graph_A_file)
        B = load_ThirdOrderSymTensor(graph_B_file)
    else

        A = load(graph_A_file,false,"COOTen")
        B = load(graph_B_file,false,"COOTen")
    end

	return align_tensors(A,B;kwargs...)

end

function get_TAME_ranks(graph_A_file::String,graph_B_file::String)

    A = load_ThirdOrderSymTensor(graph_A_file)
    B = load_ThirdOrderSymTensor(graph_B_file)

    results = Dict()

    max_iter = 30
    tol = 1e-12
    X_0 = ones(A.n,B.n)
    X_0 ./=norm(X_0)
    #V_0 = ones(B.n,1)

   # U_0 ./= norm(U_0)
    #V_0 ./= norm(V_0)

    alphas = [1.0,.5]
    betas =[0.0,1.0,10]

    for α in alphas
        for β in betas
            experiment_string = "α:$(α)_β:$(β)"
            #_,ranks = lowest_rank_TAME_original_svd_for_ranks(A,B,U_0,V_0,β, max_iter,tol,α)
            _,_,profile = TAME(A,B,X_0,β, max_iter,tol,α;profile=true)
            results[experiment_string] = profile
            println("finished $(split(graph_A_file,"/")[end])--$(split(graph_B_file,"/")[end]):$experiment_string")
        end
    end

    return results
end

#TODO: put in kwargs
function pairwise_alignment(dir)

    #align all .ssten files
    ssten_files = sort([f for f in readdir(dir) if occursin(".ssten",f)])

    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files),length(ssten_files))

    for i in 1:length(ssten_files)
        for j in i+1:length(ssten_files)
            Best_alignment_ratio[i,j] = align_from_files(dir*ssten_files[i],dir*ssten_files[j])
            Best_alignment_ratio[i,j] = Best_alignment_ratio[j,i]
        end
    end

    return ssten_files, Best_alignment_ratio
end


function distributed_pairwise_alignment(dir::String;kwargs...)

    #align all .ssten files
    ssten_files = sort([f for f in readdir(dir) if occursin(".ssten",f)])
    distributed_pairwise_alignment(ssten_files,dir;kwargs...)

end

function distributed_pairwise_alignment(files::Array{String,1},dirpath::String;
                                        method="LambdaTAME",kwargs...)

    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")

    futures = []
	results = []

    if method == "LambdaTAME"
        exp_results = zeros(Float64,length(files),length(files),3)
    else
        exp_results = zeros(Float64,length(files),length(files),4)
    end


#    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files),length(ssten_files))

    for i in 1:length(files)
        for j in i+1:length(files)

            #TODO: make this more robust
            future = @spawn align_tensors(dirpath*files[i],dirpath*files[j];kwargs...)
            push!(futures,((i,j),future))
        end
    end

    for ((i,j), future) in futures

		if method == "LambdaTAME" ||  method == "LowRankTAME"
			matched_tris, max_tris, _, _, results = fetch(future)
		elseif method == "TAME"
			matched_tris, max_tris, _, results = fetch(future)
		end
		push!(results,(files[i],files[j],matched_tris, max_tris, results))
        #TODO:update this code
        if method=="LambdaTAME"


            ratio, TAME_timings, Krylov_timings = fetch(future)
            exp_results[i,j,1] = ratio
            exp_results[j,i,1] = ratio
            exp_results[i,j,2] = TAME_timings
            exp_results[j,i,2] = TAME_timings
            exp_results[i,j,3] = Krylov_timings
            exp_results[j,i,3] = Krylov_timings
        else
            matched_tris, max_tris, total_triangles, TAME_timing = fetch(future)
            exp_results[i,j,1] = matched_tris
            exp_results[j,i,1] = matched_tris

            exp_results[i,j,2] = max_tris
            exp_results[j,i,2] = max_tris

            exp_results[i,j,3] = total_triangles
            exp_results[j,i,3] = total_triangles

            exp_results[i,j,4] = TAME_timing
            exp_results[j,i,4] = TAME_timing
        end
    end

    return files, exp_results
end


function distributed_TAME_rank_experiment(files::Array{String,1},dirpath::String;kwargs...)

    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")


    futures = []
    exp_results = []

#    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files),length(ssten_files))

    for i in 1:length(files)
        for j in i+1:length(files)

            #TODO: make this more robust
            future = @spawn get_TAME_ranks(dirpath*files[i],dirpath*files[j];kwargs...)
            push!(futures,((i,j),future))
        end
    end

    for ((i,j), future) in futures
        push!(exp_results,(files[i],files[j],fetch(future)))
    end

    return files, exp_results
end


function self_alignment(dir::String;kwargs...)

    #align all .ssten files
    ssten_files = sort([dir*"/"*f for f in readdir(dir) if occursin(".ssten",f)])
    return self_alignment(ssten_files;kwargs...)

end

#=------------------------------------------------------------------------------
                      Generated Graph Experiments Routines
------------------------------------------------------------------------------=#

function distributed_random_trials(trial_count::Int,process_count::Int,seed_exps::Bool=false
                                   ;method="LambdaTAME",graph_type::String="ER",
								   n_sizes = [10, 50, 100, 500, 1000, 5000, 10000],
								   kwargs...)

    #only handling even batch sizes
    @assert trial_count % process_count == 0

    #ensure file is loaded on all processes
    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")

    p_remove = [.01,.05]
	batches = Int(trial_count/process_count)

	if seed_exps
		Random.seed!(0)
		seeds = rand(UInt64, length(p_remove),length(n_sizes), trial_count)
	end
	seed = nothing
	results = []
    p_index = 1

    for p in p_remove

        n_index = 1

        for n in n_sizes


            for batch in 1:batches

                futures = []

                for i in 1:process_count

					if seed_exps
						seed = seeds[p_index,n_index,(batch-1)*batches + i]
					end

                    if graph_type == "ER"
                        future = @spawn full_ER_TAME_test(n,p;seed =seed,method = method,profile=true,kwargs...)
                    elseif graph_type == "HyperKron"
                        future = @spawn full_HyperKron_TAME_test(n,p;seed =seed,method = method,profile=true,kwargs...)
					elseif graph_type == "RandomGeometric"
    					future = @spawn full_Random_Geometric_Graph_TAME_test(n,p;seed =seed,method = method,profile=true,kwargs...)
                    else
                        error("invalid graph type: $graph_type\n must be either 'ER','RandomGeometric' or 'HyperKron'")
                    end
                    push!(futures,(i,seed,p,n,future))
                end

                for (i,seed,p,n,future) in futures

					if method == "LambdaTAME" ||  method == "LowRankTAME"
						matched_tris, max_tris, _, _, exp_results = fetch(future)
					elseif method == "TAME"
    					matched_tris, max_tris, _, exp_results = fetch(future)
					end
					push!(results,(i, seed, p, n, matched_tris, max_tris, exp_results))
                end
            end

            n_index += 1
        end
        p_index += 1
    end

    return results

end


function full_ER_TAME_test(n::Int,p_remove::Float64;seed=nothing,kwargs...)

	if seed !== nothing
		Random.seed!(seed)
	end

    p = 2*log(n)/n
    p_add = p*p_remove/(1-p)
    A, B = synthetic_erdos_problem(n,p,p_remove,p_add)

	return set_up_tensor_alignment(A, B;kwargs...)
end

function full_HyperKron_TAME_test(n::Int,p_remove::Float64;seed=nothing,kwargs...)

	if seed !== nothing
		Random.seed!(seed)
	end

    p = .4#2*log(n)/n
    r = .4
    p_add = p*p_remove/(1-p)
    A, B = synthetic_HyperKron_problem(n,p,r,p_remove,p_add)

	return set_up_tensor_alignment(A, B;kwargs...)
end

function full_Random_Geometric_Graph_TAME_test(n::Int,p_remove::Float64;seed=nothing,kwargs...)

	if seed !== nothing
		Random.seed!(seed)
	end

    k = 10
	p = k/n
    p_add = p*p_remove/(1-p)
    A, B = synthetic_Random_Geometric_problem(n,k,p_remove,p_add)

	return set_up_tensor_alignment(A, B;kwargs...)
end

function set_up_tensor_alignment(A::SparseMatrixCSC{Float64,Int64},B::SparseMatrixCSC{Float64,Int64};kwargs...)

	n,n = size(A)
	#permute the rows of B
    perm = shuffle(1:n)
    B = B[perm,perm]

    #build COOTens from graphs
    A_tris = collect(MatrixNetworks.triangles(A))
    B_tris = collect(MatrixNetworks.triangles(B))

    A_nnz = length(A_tris)
    B_nnz = length(B_tris)

    A_indices = Array{Int,2}(undef,A_nnz,3)
    B_indices = Array{Int,2}(undef,B_nnz,3)
    for i =1:A_nnz
        A_indices[i,1]= A_tris[i][1]
        A_indices[i,2]= A_tris[i][2]
        A_indices[i,3]= A_tris[i][3]
    end

    for i =1:B_nnz
        B_indices[i,1]= B_tris[i][1]
        B_indices[i,2]= B_tris[i][2]
        B_indices[i,3]= B_tris[i][3]
    end

    A_vals = ones(A_nnz)
    B_vals = ones(B_nnz)

    A_ten = ThirdOrderSymTensor(n,A_indices,A_vals)
    B_ten = ThirdOrderSymTensor(n,B_indices,B_vals)
    return align_tensors(A_ten,B_ten;kwargs...)
end



"""-----------------------------------------------------------------------------
    synthetic_erdos_problem(n,p,p_remove,p_add)

    Creates two graphs to be aligned against one another to enumerate triangles
  between the two graphs.

Inputs:
-------
* n - (Int):
    the number of nodes in the graphs.

* p - (float):
    The probability of including an edge in the original graph.

* p_remove - (float):
    The probability of removing an edge from the original graph to produce the
    second graph.
* p_add - (float):
    The probability of adding in a new edge in the new graph.
-----------------------------------------------------------------------------"""
function synthetic_erdos_problem(n,p,p_remove,p_add)
    A = erdos_renyi(n,p)
    B = copy(A)


    is,js,_ = findnz(erdos_renyi(n,p_remove))
    for (i,j) in zip(is,js)
        if A[i,j] > 0
            B[i,j] = 0
            B[j,i] = 0
        end
    end

    is,js,_ = findnz(erdos_renyi(n,p_add))
    for (i,j) in zip(is,js)
        if A[i,j] == 0.0
            B[i,j] = 1
            B[j,i] = 1
        end
    end

    return A, B
end

function erdos_renyi(n,p)
    A = sprand(n,n,p)
    U = triu(A,1)
    is,js,_ = findnz(max.(U,U'))
    return sparse(is,js,ones(length(is)),n,n)
end

"""-----------------------------------------------------------------------------
    synthetic_erdos_problem(n,p,p_remove,p_add)

    Creates two graphs to be aligned against one another to enumerate triangles
  between the two graphs.

Inputs:
-------
* n - (Int):
    the number of nodes in the graphs.

* p - (float):
    The probability of including an edge in the original graph.

* p_remove - (float):
    The probability of removing an edge from the original graph to produce the
    second graph.
* p_add - (float):
    The probability of adding in a new edge in the new graph.
-----------------------------------------------------------------------------"""
function synthetic_HyperKron_problem(n,p,r,p_remove,p_add)

    A = sparse(gpa_graph(n,p,r,5))
    B = copy(A)


    is,js,_ = findnz(erdos_renyi(n,p_remove))
    for (i,j) in zip(is,js)
        if A[i,j] > 0
            B[i,j] = 0
            B[j,i] = 0
        end
    end

    is,js,_ = findnz(erdos_renyi(n,p_add))
    for (i,j) in zip(is,js)
        if A[i,j] == 0.0
            B[i,j] = 1
            B[j,i] = 1
        end
    end

    return A, B
end


"""-----------------------------------------------------------------------------
    synthetic_Random_Geometric_problem(n,p,p_remove,p_add)

    Creates two graphs to be aligned against one another to enumerate triangles
  between the two graphs. First graph is created using a random geometric graph,
  with a given n, and number of nearest neighbors k. The second graph is created
  by

Inputs:
-------
* n - (Int):
    the number of nodes in the graphs.

* k - (float):
    The number of nearest neighbors to connected in the graph.

* p_remove - (float):
    The probability of removing an edge from the original graph to produce the
    second graph.

* p_add - (float):
    The probability of adding in a new edge in the new graph.
-----------------------------------------------------------------------------"""
function synthetic_Random_Geometric_problem(n,k,p_remove,p_add)
	_, ei, ej = random_geometric_graph(n,k)
    C = sparse(ei,ej,1.0,n,n)
	is,js,_ = findnz(max.(C,C')) #symmetrize output

	A = sparse(is,js,1.0,n,n)
    B = copy(A)

    is,js,_ = findnz(erdos_renyi(n,p_remove))
    for (i,j) in zip(is,js)
        if A[i,j] > 0
            B[i,j] = 0
            B[j,i] = 0
        end
    end

    is,js,_ = findnz(erdos_renyi(n,p_add))
    for (i,j) in zip(is,js)
        if A[i,j] == 0.0
            B[i,j] = 1
            B[j,i] = 1
        end
    end

    return A, B
end

function random_geometric_graph(n,k)
  xy = rand(2,n)
  T = BallTree(xy)
  idxs = knn(T, xy, k)[1]
  # form the edges for sparse
  ei = Int[]
  ej = Int[]
  for i=1:n
    for j=idxs[i]
      if i != j
        push!(ei,i)
        push!(ej,j)
      end
    end
  end
  return xy, ei, ej
end
