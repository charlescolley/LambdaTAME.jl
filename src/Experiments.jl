#=------------------------------------------------------------------------------
                        Local File Experiments Routines
------------------------------------------------------------------------------=#
function self_alignment(ssten_files::Array{String,1})

    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files))

    for f in ssten_files
        A = ssten.load(f,false,"COOTen")

        #permute tensor to align against
        p = sort(1:A.cubical_dimension,by=i->rand())
        A_permuted = ssten.COOTen(A.indices,A.vals,A.cubical_dimension)
        ssten.permute_tensor!(A_permuted,p)

        best_score, _, _ = align_tensors(A,A_permuted)

        println("$f aligned with $best_score")
    end


end

function align_tensors(graph_A_file::String,graph_B_file::String)

    A = ssten.load(graph_A_file,false,"COOTen")
    B = ssten.load(graph_B_file,false,"COOTen")

    return align_tensors(A,B)
end

function pairwise_alignment(dir)

    #align all .ssten files
    ssten_files = sort([f for f in readdir(dir) if occursin(".ssten",f)])

    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files),length(ssten_files))

    for i in 1:length(ssten_files)
        for j in i+1:length(ssten_files)
            Best_alignment_ratio[i,j] = align_from_files(dir*"/"*ssten_files[i],dir*"/"*ssten_files[j])
            Best_alignment_ratio[i,j] = Best_alignment_ratio[j,i]
        end
    end

    return ssten_files, Best_alignment_ratio
end

function distributed_pairwise_alignment(dir)


    @everywhere include_string(Main,$(read("TAME++.jl",String)),"TAME++.jl")
    #align all .ssten files
    ssten_files = sort([f for f in readdir(dir) if occursin(".ssten",f)])

    futures = []
    exp_results = zeros(Float64,length(ssten_files),length(ssten_files),3)
#    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files),length(ssten_files))

    for i in 1:length(ssten_files)
        for j in i+1:length(ssten_files)

            future = @spawn align_tensors(dir*"/"*ssten_files[i],dir*"/"*ssten_files[j])
            push!(futures,((i,j),future))
        end
    end

    for ((i,j), future) in futures

        ratio, TAME_timings, Krylov_timings = fetch(future)
        exp_results[i,j,1] = ratio
        exp_results[j,i,1] = ratio
        exp_results[i,j,2] = TAME_timings
        exp_results[j,i,2] = TAME_timings
        exp_results[i,j,3] = Krylov_timings
        exp_results[j,i,3] = Krylov_timings
    end

    return ssten_files, exp_results
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
    @everywhere include_string(Main,$(read("LambdaTAMEjl",String)),"LambdaTAME.jl")

    n_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    p_remove = [.01,.05]

    exp_results = zeros(Float64,length(p_remove),length(n_sizes),trial_count,5)

    p_index = 1

    for p in p_remove

        n_index = 1

        for n in n_sizes


            for batch in 1:(trial_count/process_count)
                futures = []

                for i in 1:trial_count

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

    A_ten = ssten.COOTen(A_indices,A_vals,n)
    B_ten = ssten.COOTen(B_indices,B_vals,n)
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

    A_ten = ssten.COOTen(A_indices,A_vals,n)
    B_ten = ssten.COOTen(B_indices,B_vals,n)
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