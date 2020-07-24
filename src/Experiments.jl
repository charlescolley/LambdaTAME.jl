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
                       ThirdOrderSparse=true,method="LambdaTAME")

    if ThirdOrderSparse
        A = load_ThirdOrderSymTensor(graph_A_file)
        B = load_ThirdOrderSymTensor(graph_B_file)
    else

        A = load(graph_A_file,false,"COOTen")
        B = load(graph_B_file,false,"COOTen")
    end

    if method == "LambdaTAME"
        return align_tensors(A,B)
    elseif method =="LowRankTAME"
        return align_tensors_with_TAME(A,B)
    elseif method == "TAME"
        return align_tensors_with_TAME(A,B;low_rank=false)
    else
        error("method must be one of 'LambdaTAME','LowRankTAME', or 'TAME'.")
    end

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
    betas =[0.0,1.0]

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

function distributed_pairwise_alignment(dir::String,method="LambdaTAME")

    #align all .ssten files
    ssten_files = sort([f for f in readdir(dir) if occursin(".ssten",f)])
    distributed_pairwise_alignment(ssten_files,dir ,method)

end

function distributed_pairwise_alignment(files::Array{String,1},dirpath::String;method="LambdaTAME")

    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")

    futures = []

    if method == "LambdaTAME"
        exp_results = zeros(Float64,length(files),length(files),3)
    else
        exp_results = zeros(Float64,length(files),length(files),4)
    end


#    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files),length(ssten_files))

    for i in 1:length(files)
        for j in i+1:length(files)

            #TODO: make this more robust
            future = @spawn align_tensors(dirpath*files[i],dirpath*files[j],method=method)
            push!(futures,((i,j),future))
        end
    end

    for ((i,j), future) in futures

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


function distributed_TAME_rank_experiment(files::Array{String,1},dirpath::String)#;method="LambdaTAME")

    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")

    futures = []
    exp_results = []

#    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files),length(ssten_files))

    for i in 1:length(files)
        for j in i+1:length(files)

            #TODO: make this more robust
            future = @spawn get_TAME_ranks(dirpath*files[i],dirpath*files[j])
            push!(futures,((i,j),future))
        end
    end

    for ((i,j), future) in futures
        push!(exp_results,(files[i],files[j],fetch(future)))
    end

    return files, exp_results
end


function self_alignment(dir::String)

    #align all .ssten files
    ssten_files = sort([dir*"/"*f for f in readdir(dir) if occursin(".ssten",f)])
    return self_alignment(ssten_files)

end

#=------------------------------------------------------------------------------
                      Generated Graph Experiments Routines
------------------------------------------------------------------------------=#

function distributed_random_trials(trial_count::Int,process_count::Int,graph_type::String="ER")

    #only handling even batch sizes
    @assert trial_count % process_count == 0

    #ensure file is loaded on all processes
    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")

    n_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    p_remove = [.01,.05]

    exp_results = zeros(Float64,length(p_remove),length(n_sizes),trial_count,5)

    p_index = 1

    for p in p_remove

        n_index = 1

        for n in n_sizes


            for batch in 1:Int(trial_count/process_count)

                futures = []

                for i in 1:process_count

                    if graph_type == "ER"
                        future = @spawn full_ER_TAME_test(n,p)
                    elseif graph_type == "HyperKron"
                        future = @spawn full_HyperKron_TAME_test(n,p)
                    else
                        error("invalid graph type: $graph_type\n must be either ER or HyperKron")
                    end
                    push!(futures,(i,future))
                end

                for (i,future) in futures
                    exp_index = (batch-1)*process_count + i
                    mached_tris, max_tris, total_triangles, TAME_time, Krylov_time = fetch(future)

                    exp_results[p_index,n_index,exp_index,1] = mached_tris
                    exp_results[p_index,n_index,exp_index,2] = max_tris
                    exp_results[p_index,n_index,exp_index,3] = total_triangles
                    exp_results[p_index,n_index,exp_index,4] = TAME_time
                    exp_results[p_index,n_index,exp_index,5] = Krylov_time

                end
            end

#            exp_results[p_index,n_index,1] /= trial_count
#            exp_results[p_index,n_index,2] /= trial_count
#            exp_results[p_index,n_index,3] /= trial_count

            n_index += 1
        end
        p_index += 1
    end

    return exp_results

end


function full_ER_TAME_test(n::Int,p_remove::Float64)

    p = 2*log(n)/n
    p_add = p*p_remove/(1-p)
    A, B = synthetic_erdos_problem(n,p,p_remove,p_add)

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

    A_ten = COOTen(A_indices,A_vals,n)
    B_ten = COOTen(B_indices,B_vals,n)
    return align_tensors(A_ten,B_ten)

end

function full_HyperKron_TAME_test(n::Int,p_remove::Float64)

    p = .4#2*log(n)/n
    r = .4
    p_add = p*p_remove/(1-p)
    A, B = synthetic_HyperKron_problem(n,p,r,p_remove,p_add)

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

    A_ten = COOTen(A_indices,A_vals,n)
    B_ten = COOTen(B_indices,B_vals,n)
    return align_tensors(A_ten,B_ten)
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