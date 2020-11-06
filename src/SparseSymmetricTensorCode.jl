SST_PATH = "/Users/ccolley/Documents/Research/SparseSymmetricTensors.jl/src/SparseSymmetricTensors.jl" # local path
#SST_PATH = "/homes/ccolley/Documents/Software/SparseSymmetricTensors.jl/src/SparseSymmetricTensors.jl" #Nilpotent path
include(SST_PATH)
using Main.SparseSymmetricTensors



#===============================================================================
                            Contraction Codes
===============================================================================#

#TODO: adapt to work for any order tensor
function implicit_contraction(A::COOTen,B::COOTen,x::Array{Float64,1})

    @assert length(x) == A.cubical_dimension*B.cubical_dimension
    m = A.cubical_dimension
    n = B.cubical_dimension
    y = similar(x)

	y .= 0

    ileave = (i,j) -> i + m*(j-1)

    for i in 1:length(A)

        for (i_1,i_2,i_3) in permutations(A.indices[i,:])

            for j in 1:length(B)
                j_1,j_2,j_3 = B.indices[j,:]

                y[ileave(i_1,j_1)] += 2*A.vals[i]*B.vals[j]*x[ileave(i_2,j_2)]*x[ileave(i_3,j_3)]
                y[ileave(i_2,j_2)] += 2*A.vals[i]*B.vals[j]*x[ileave(i_1,j_1)]*x[ileave(i_3,j_3)]
                y[ileave(i_3,j_3)] += 2*A.vals[i]*B.vals[j]*x[ileave(i_1,j_1)]*x[ileave(i_2,j_2)]

            end
        end
    end

    return y
end


#===============================================================================
                            Matching Code
===============================================================================#

#Computes the TAME score for this iterate by
function TAME_score(A::COOTen,B::COOTen,u::Array{Float64,1},v::Array{Float64,1})

    Match_mapping, weight = rank_one_matching(u,v)
    TAME_score(A,B,Match_mapping)

end

function TAME_score(A::COOTen,B::COOTen,U::Array{Float64,2},V::Array{Float64,2})

    Match_mapping = low_rank_matching(U,V)
    TAME_score(A,B,Match_mapping)

end

function TAME_score(A::COOTen,B::COOTen,X::Array{Float64,2};return_timings=false)

    if return_timings
         (_,_,matching,_) ,scoring_time = @timed bipartite_matching_primal_dual(X)
         (triangle_count, gaped_triangles,inverted_matching), matching_time = @timed TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
         return triangle_count, gaped_triangles,inverted_matching, matching_time, matching_time
     else
         (_,_,matching,_) ,scoring_time = @timed bipartite_matching_primal_dual(X)
         return TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
     end
 
 end
 
 function TAME_score(A::COOTen,B::COOTen,x::Array{Float64,1};return_timings=false)
 
    if return_timings
         (_,_,matching) ,matching_time = @timed bipartite_matching_primal_dual(reshape(x,A.cubical_dimension,B.cubical_dimension))
         (triangle_count, gaped_triangles,inverted_matching), scoring_time = @timed TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
         return triangle_count, gaped_triangles,inverted_matching, matching_time,scoring_time
     else
         _,_,matching = bipartite_matching_primal_dual(reshape(x,A.cubical_dimension,B.cubical_dimension))
         return TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
     end
 
 end


function TAME_score(A::COOTen,B::COOTen,Match_mapping::Dict{Int,Int})

    match_len = length(Match_mapping)

    Triangle_check = Dict{Array{Int,1},Int}()
    gaped_triangles = 0
    triangle_count = 0

    if A.unique_nnz > B.unique_nnz

        for i in 1:A.unique_nnz
            Triangle_check[A.indices[i,:]] = 1
        end

        #invert to map v indices to u
        Match_mapping = Dict(value => key for (key, value) in Match_mapping)

        for i in 1:B.unique_nnz
            v_i,v_j,v_k = B.indices[i,:]

            matched_triangle =
              sort([get(Match_mapping,v_i,-1),get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])
    #        println(B.indices[i,:]," -> ",matched_triangle)
            match = get(Triangle_check,matched_triangle,0)
            if match == 1
                triangle_count += 1
            else
                gaped_triangles += 1
            end
        end

    else
        for i in 1:B.unique_nnz
            Triangle_check[B.indices[i,:]] = 1
        end

        for i in 1:A.unique_nnz
            v_i,v_j,v_k = A.indices[i,:]
            matched_triangle =
               sort([get(Match_mapping,v_i,-1), get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])

            match = get(Triangle_check,matched_triangle,0)
            if match == 1
                triangle_count += 1
            else
                gaped_triangles += 1
            end
        end
    end

   # triangles, triangle_weight = count_triangles(sub_A,sub_B)

    return triangle_count, gaped_triangles, Match_mapping

end

function TAME_score(Triangle_Dict::Dict{Array{Int,1},Int},Input_tensor::COOTen,
    Match_mapping::Dict{Int,Int})

    triangle_count = 0
    gaped_triangles = 0

    for i in 1:Input_tensor.unique_nnz
        v_i,v_j,v_k = Input_tensor.indices[i,:]

        matched_triangle =
            sort([get(Match_mapping,v_i,-1),get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])

        match = get(Triangle_Dict,matched_triangle,0)
        if match == 1
            triangle_count += 1
        else
            gaped_triangles += 1
        end
    end
    return triangle_count, gaped_triangles, Match_mapping
end


function search_Krylov_space(A::COOTen,B::COOTen,U::Array{Float64,2},V::Array{Float64,2})

    best_score = -1
    best_i = -1
    best_j = -1
    best_matching = Dict{Int,Int}()

    Triangle_check = Dict{Array{Int,1},Int}()

    if A.unique_nnz > B.unique_nnz
        for i in 1:A.unique_nnz
            Triangle_check[A.indices[i,:]] = 1
        end
        Input_tensor = B
    else
        for i in 1:B.unique_nnz
            Triangle_check[B.indices[i,:]] = 1
        end
        Input_tensor = A
    end

    for i in 1:size(U,2)
       for j in 1:size(V,2)

            if A.unique_nnz > B.unique_nnz
                matched_tris, gaped_tris, matching = TAME_score(Triangle_check,Input_tensor,V[:,j],U[:,i])
            else
                matched_tris, gaped_tris, matching = TAME_score(Triangle_check,Input_tensor,U[:,i],V[:,j])
            end

            if matched_tris > best_score
                best_matching = matching
                best_score = matched_tris
                best_i = i
                best_j = j
            end
        end
    end
    return best_score, best_i, best_j, best_matching
end

#===============================================================================
                          TAME Implementations Code
===============================================================================#

#
#  Parameter Searching Code
#


function align_tensors(A::COOTen,B::COOTen;method::String="LambdaTAME",
                       no_matching=false,kwargs...)

    #put larger tensor on the left
    if B.n > A.n
        results = align_tensors(B,A;method = method, no_matching=no_matching,kwargs...)
        #flip the matchings if A and B were swapped
        if method == "LambdaTAME" ||  method == "LowRankTAME"
            best_TAME_PP_tris, max_triangle_match, U_best, V_best, best_matching = results
            return best_TAME_PP_tris, max_triangle_match, U_best, V_best, Dict((j,i) for (i,j) in best_matching)
        elseif method == "TAME"
            best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, best_matching = results
            return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, Dict((j,i) for (i,j) in best_matching)
        end
    end

    if method == "LambdaTAME"
        return ΛTAME_param_search(A,B;kwargs...)
    elseif method == "LowRankTAME"
        return LowRankTAME_param_search(A,B;no_matching = no_matching,kwargs...)
    elseif method == "TAME"
        return TAME_param_search(A,B;no_matching = no_matching,kwargs...)
    else
        throw(ArgumentError("method must be one of 'LambdaTAME', 'LowRankTAME',or 'TAME'."))
    end
end

function align_tensors_profiled(A::COOTen,B::COOTen; method::String="LambdaTAME",
                                no_matching=false,kwargs...)

    #put larger tensor on the left
    if B.n > A.n
        results =  align_tensors_profiled(B,A;method = method, no_matching=no_matching,kwargs...)
        #flip the matchings if A and B were swapped
        if method == "LambdaTAME" ||  method == "LowRankTAME"
            best_TAME_PP_tris, max_triangle_match, U_best, V_best, best_matching,profile = results
            return best_TAME_PP_tris, max_triangle_match, U_best, V_best, Dict((j,i) for (i,j) in best_matching), profile
        elseif method == "TAME"
            best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, best_matching,profile = results
            return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, Dict((j,i) for (i,j) in best_matching), profile
        end
    end

    if method == "LambdaTAME"
        return ΛTAME_param_search_profiled(A,B;kwargs...)
    elseif method == "LowRankTAME"
        return LowRankTAME_param_search_profiled(A,B;no_matching = no_matching,kwargs...)
    elseif method == "TAME"
        return TAME_param_search_profiled(A,B;no_matching = no_matching,kwargs...)
    else
        throw(ArgumentError("method must be one of 'LambdaTAME', 'LowRankTAME', or 'TAME'."))
    end
end


function ΛTAME_param_search_profiled(A::COOTen,B::COOTen; iter::Int = 15,tol::Float64=1e-6,
                                     alphas::Array{F,1}=[.5,1.0],
                                     betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001]) where {F <: AbstractFloat}

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    best_TAME_PP_tris = -1
    best_i  = -1
    best_j = -1
    best_matching = Dict{Int,Int}()

    if Ten == COOTen
        m = A.cubical_dimension
        n = B.cubical_dimension
    else
        m = A.n
        n = B.n
    end


    results = Dict(
        "TAME_timings" => Array{Float64,1}(undef,length(alphas)*length(betas)),
        "Krylov Timings"=> Array{Float64,1}(undef,length(alphas)*length(betas))
    )
    exp_index = 1

    U = Array{Float64,2}(undef,m,iter)
    V = Array{Float64,2}(undef,n,iter)


    for α in alphas
        for beta in betas

            ((U,V),runtime) = @timed ΛTAME(A,B,beta,iter,tol,α)
            results["TAME_timings"][exp_index] = runtime

            #search the Krylov Subspace
            ((search_tris, i, j, matching),runtime) = @timed search_Krylov_space(A,B,U,V)
            results["Krylov Timings"][exp_index] = runtime
            exp_index += 1


            if search_tris > best_TAME_PP_tris
                best_matching = matching
                best_TAME_PP_tris = search_tris
                best_i = i
                best_j = j
            end

            println("α:$(α) -- β:$(beta) finished -- tri_match:$search_tris -- max_tris $(max_triangle_match) -- best tri_match: $best_TAME_PP_tris")
        end
    end

    println("best i:$best_i -- best j:$best_j")
    return best_TAME_PP_tris, max_triangle_match, U[:,best_i], V[:,best_j], best_matching, results

end

#add in SparseSymmetricTensors.jl function definitions
function TAME_param_search(A::COOTen,B::COOTen;iter::Int = 15,tol::Float64=1e-6,
                           alphas::Array{F,1}=[.5,1.0],
                           betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
                           kwargs...) where {F <: AbstractFloat}

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)
    best_TAME_PP_tris::Int = -1
    best_matching = Dict{Int,Int}()


    m = A.cubical_dimension
    n = B.cubical_dimension


    best_TAME_PP_x = Array{Float64,2}(undef,m,n)


    for α in alphas
        for β in betas

            x, triangle_count, matching = TAME(A,B,β,iter,tol,α;W = ones(m,n),kwargs...)

            if triangle_count > best_TAME_PP_tris
                best_matching = matching
                best_TAME_PP_tris = triangle_count
                best_TAME_PP_x = copy(x)
            end
            println("α:$(α) -- β:$β finished -- tri_match:$(triangle_count) -- max_tris $(max_triangle_match) -- best tri_match:$(best_TAME_PP_tris)")
        end

    end

    return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, best_matching
end

function TAME_param_search_profiled(A::COOTen,B::COOTen;iter::Int = 15,tol::Float64=1e-6,
                                    alphas::Array{F,1}=[.5,1.0],
                                    betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
                                    profile::Bool=false,profile_aggregation="all",
                                    kwargs...) where {F <: AbstractFloat}
    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)
    best_TAME_PP_tris = -1
    best_matching = Dict{Int,Int}()

    m = A.cubical_dimension
    n = B.cubical_dimension


    best_TAME_PP_x = Array{Float64,2}(undef,m,n)
    experiment_profiles = Array{Tuple{String,Dict{String,Union{Array{Float64,1},Array{Array{Float64,1},1}}}},1}(undef,0)

    for α in alphas
        for β in betas

            x, triangle_count, matching, experiment_profile = TAME_profiled(A,B,β,iter,tol,α;W = ones(m,n),kwargs...)
            push!(experiment_profiles,("α:$(α)_β:$(β)",experiment_profile))

            if triangle_count > best_TAME_PP_tris
                best_matching = matching
                best_TAME_PP_tris = triangle_count
                best_TAME_PP_x = copy(x)
            end
            println("α:$(α) -- β:$β finished -- tri_match:$(best_TAME_PP_tris) -- max_tris $(max_triangle_match)")
        end
    end

    return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, best_matching, experiment_profiles

end

function LowRankTAME_param_search(A::COOTen,B::COOTen;iter::Int = 15,tol::Float64=1e-6,
                                  U_0::Array{Float64,2} = ones(A.n,1), V_0::Array{Float64,2} = ones(B.n,1),
                                  alphas::Array{F,1}=[.5,1.0],
                                  betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
                                  kwargs...) where {F <: AbstractFloat}

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)
    best_TAME_PP_tris = -1
    best_matching = Dict{Int,Int}()

    m = A.cubical_dimension
    n = B.cubical_dimension

    best_TAME_PP_U = ones(m,1)
    best_TAME_PP_V = ones(n,1)

    for α in alphas
        for β in betas
            U, V, triangle_count,matching =
            LowRankTAME(A,B,U_0,V_0,β,iter,tol,α;kwargs...)

            if triangle_count > best_TAME_PP_tris
                best_TAME_PP_tris = triangle_count
                best_matching = matching
                best_TAME_PP_U = copy(U)
                best_TAME_PP_V = copy(V)
            end
            println("α:$(α) -- β:$(β) -- tri_match:$(triangle_count) -- max_tris:$(max_triangle_match) -- best tri match:$best_TAME_PP_tris")
        end
    end

    return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_U, best_TAME_PP_V, best_matching

end

#
#  Spectral Relaxation Code
#

function ΛTAME(A::COOTen, B::COOTen, β::Float64, max_iter::Int,
    tol::Float64,α::Float64;update_user::Int=-1)

    U = zeros(A.cubical_dimension,max_iter+1)
    V = zeros(B.cubical_dimension,max_iter+1) #store initial in first column

    U[:,1] = ones(A.cubical_dimension)
    U[:,1] /=norm(U[:,1])

    V[:,1] = ones(B.cubical_dimension)
    V[:,1] /=norm(U[:,1])

    sqrt_β = β^(.5)

    lambda = Inf
    i = 1

    while true

        U[:,i+1] = contract_k_1(A,U[:,i])
        V[:,i+1] = contract_k_1(B,V[:,i])

        lambda_A = (U[:,i+1]'*U[:,i])
        lambda_B = (V[:,i+1]'*V[:,i])
        new_lambda = lambda_A*lambda_B

        if β != 0.0
            U[:,i+1] .+= sqrt_β*U[:,i+1]
            V[:,i+1] .+= sqrt_β*V[:,i+1]
        end

        if α != 1.0
            U[:,i+1] = α*U[:,i+1] + (1 -α)*U[:,1]
            V[:,i+1] = α*V[:,i+1] + (1 -α)*V[:,1]
        end

        U[:,i+1] ./= norm(U[:,i+1])
        V[:,i+1] ./= norm(V[:,i+1])

        if update_user != -1 && i % update_user == 0
            println("iteration $(i)    λ_A: $(lambda_A) -- λ_B: $(lambda_B) -- newλ: $(new_lambda)")
        end

        if abs(new_lambda - lambda) < tol || i >= max_iter
            return U[:,1:i], V[:,1:i]
        else
            lambda = new_lambda
            i += 1
        end
    end
end

#=
  TODO: currently missing COOTen Implementations for 
   - LowRankTAME
   - LowRankTAME_profiled
   - TAME
   - TAME_profiled
=#

#===============================================================================
                            PostProcessing Code
===============================================================================#
"""-----------------------------------------------------------------------------
   Computes a post processing routine on the optimal aligment produced by
   Λ-TAME. The algorithm runs a b-matching to find suitable replacements
   and computes whether or not to make the swap if the number of aligned
   triangles increases.

   Inputs:
   -------
   {A,B} - (COOTen)
     Coordinate representations of adjacency tensors for the motif induced
     hypergraphs.
   {u,v} - (Array{Float64,1})
     The components to the rank-1 alignment matrix which offered the best
     alignment scores.
   iterations - (Int)
     The number of iterations to run the post-processing algorithm for.
   b - (Int)
     The size of the b-matching.

-----------------------------------------------------------------------------"""
function post_process(A::COOTen,B::COOTen,u::Array{Float64,1},
                      v::Array{Float64,1},iterations::Int,b::Int)

    @assert length(u) <= length(v)

    Matching = rank_one_matching(u,v)

    A_incidence = produce_triangle_incidence(A.indices,A.cubical_dimension)
    B_incidence = produce_triangle_incidence(B.indices,B.cubical_dimension)

    potential_matchings = b_matching(u,v,b)

    u_sum = sum(u)
    v_sum = sum(v)

    match_q = Queue{Tuple{Int,Int,Float64}}()

    for _ in 1:iterations
        #sort by ∑_i x(ii') + ∑_i' x(ii')
        [enqueue!(q,z) for z in sort(potential_matchings,by=x->u[x[1]]*u_sum+ v[x[2]]*v_sum)]
        while length(q) != 0

            i,ip,_ = dequeue!(match_q)

            matched_tris = compute_triangles_aligned(A_incidence[i],B_incidence[ip],Matching)

            #TODO: optimize this later
            Pref_i = [k for (j,k,_) in potential_matchings if j == i]
            Pref_ip = [k for (j,k,_) in potential_matchings if j == ip]



        end
    end
end


