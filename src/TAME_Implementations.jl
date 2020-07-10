
#=------------------------------------------------------------------------------
              Routines for searching over alpha/beta parameters
------------------------------------------------------------------------------=#

function align_tensors(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor)
    iter =15
    tol=1e-6
    rank = 10
    alphas = [.15,.5,.85]
    betas =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001]

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)

    TAME_timings = Array{Float64,1}(undef,length(alphas)*length(betas))
    Krylov_Search_timings = Array{Float64,1}(undef,length(alphas)*length(betas))

    U = Array{Float64,2}(undef,A.n,iter)
    V = Array{Float64,2}(undef,B.n,iter)

    best_TAME_PP_tris = -1
    best_i  = -1
    best_j = -1
    exp_index = 1

    for α in alphas
        for beta in betas
            ((U,V),runtime) = @timed ΛTAME(A,B,beta,iter,tol,α)
            TAME_timings[exp_index] = runtime

            #search the Krylov Subspace
            ((search_tris, i, j),runtime) = @timed search_Krylov_space(A,B,U,V)
            Krylov_Search_timings[exp_index] = runtime

            if search_tris > best_TAME_PP_tris
                best_TAME_PP_tris = search_tris
                best_i = i
                best_j = j
            end
            exp_index += 1
        end

    end

    avg_TAME_timings = sum(TAME_timings)/length(TAME_timings)
    avg_Krylov_timings = sum(Krylov_Search_timings)/length(Krylov_Search_timings)

    return best_TAME_PP_tris,max_triangle_match,total_triangles, avg_TAME_timings, avg_Krylov_timings
end


function align_tensors(A::COOTen,B::COOTen)
    iter =15
    tol=1e-6
    rank = 10
    alphas = [.15,.5,.85]
    betas =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001]

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)

    TAME_timings = Array{Float64,1}(undef,length(alphas)*length(betas))
    Krylov_Search_timings = Array{Float64,1}(undef,length(alphas)*length(betas))

    U = Array{Float64,2}(undef,A.cubical_dimension,iter)
    V = Array{Float64,2}(undef,B.cubical_dimension,iter)

    best_TAME_PP_tris = -1
    best_i  = -1
    best_j = -1
    exp_index = 1

    for α in alphas
        for beta in betas
            ((U,V),runtime) = @timed ΛTAME(A,B,beta,iter,tol,α)
            TAME_timings[exp_index] = runtime

            #search the Krylov Subspace
            ((search_tris, i, j),runtime) = @timed search_Krylov_space(A,B,U,V)
            Krylov_Search_timings[exp_index] = runtime

            if search_tris > best_TAME_PP_tris
                best_TAME_PP_tris = search_tris
                best_i = i
                best_j = j
            end
            exp_index += 1
        end

    end

    avg_TAME_timings = sum(TAME_timings)/length(TAME_timings)
    avg_Krylov_timings = sum(Krylov_Search_timings)/length(Krylov_Search_timings)

    return best_TAME_PP_tris,max_triangle_match,total_triangles, avg_TAME_timings, avg_Krylov_timings
end

#add in SparseSymmetricTensors.jl function definitions
function align_tensors_with_TAME(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,low_rank=true)

    iter =15
    tol=1e-6
    alphas = [.15,.5,.85]
    betas =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001]

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)

    Timings = Array{Float64,1}(undef,length(alphas)*length(betas))
#    Krylov_Search_timings = Array{Float64,1}(undef,length(alphas)*length(betas))

 #   U = Array{Float64,2}(undef,A.n,iter)
 #   V = Array{Float64,2}(undef,B.n,iter)

    best_TAME_PP_tris = -1
    best_i  = -1
    best_j = -1
    exp_index = 1

    for α in alphas
        for beta in betas
            if low_rank
                (_, triangle_count),runtime =
                    @timed lowest_rank_TAME(A,B,ones(A.n,1),ones(B.n,1),
                                            beta,iter,tol,α;profile=false)
            else
                (_, triangle_count),runtime =
                    @timed TAME(A,B,ones(A.n,B.n),beta,iter,tol,α;profile=false)

            end

            Timings[exp_index] = runtime

            if triangle_count > best_TAME_PP_tris
                best_TAME_PP_tris = triangle_count
            end
            exp_index += 1
        end

    end

    avg_TAME_timings = sum(TAME_timings)/length(TAME_timings)
    avg_Krylov_timings = sum(Krylov_Search_timings)/length(Krylov_Search_timings)

    return best_TAME_PP_tris,max_triangle_match,total_triangles, avg_TAME_timings, avg_Krylov_timings

end
#=------------------------------------------------------------------------------
              Routines for searching over alpha/beta parameters
------------------------------------------------------------------------------=#

function ΛTAME(A::COOTen, B::COOTen, β::Float64, max_iter::Int,
               tol::Float64,α::Float64)

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

       # println("iteration $(i)    λ_A: $(lambda_A) -- λ_B: $(lambda_B) -- newλ: $(new_lambda)")

        if abs(new_lambda - lambda) < tol || i >= max_iter
            return U[:,1:i], V[:,1:i]
        else
            lambda = new_lambda
            i += 1
        end

    end

end

function ΛTAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor, β::Float64,
               max_iter::Int,tol::Float64,α::Float64)

    U = zeros(A.n,max_iter+1)
    V = zeros(B.n,max_iter+1) #store initial in first column

    U[:,1] = ones(A.n)
    U[:,1] /=norm(U[:,1])

    V[:,1] = ones(B.n)
    V[:,1] /=norm(U[:,1])

    sqrt_β = β^(.5)

    lambda = Inf
    i = 1

    while true

        U[:,i+1] = tensor_vector_contraction(A,U[:,i])
        V[:,i+1] = tensor_vector_contraction(B,V[:,i])

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

       # println("iteration $(i)    λ_A: $(lambda_A) -- λ_B: $(lambda_B) -- newλ: $(new_lambda)")

        if abs(new_lambda - lambda) < tol || i >= max_iter
            return U[:,1:i], V[:,1:i]
        else
            lambda = new_lambda
            i += 1
        end

    end

end


#runs TAME, but reduces down to lowest rank form first
function lowest_rank_TAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,W::Array{F,2},
                       β::F, max_iter::Int,tol::F,α::F;profile=false) where {F <:AbstractFloat}

    dimension = minimum((A.n,B.n))

	(U_k,S,VT),t = @timed svd(W)
	singular_indexes = [i for i in 1:length(S) if S[i] > S[1]*eps(Float64)*dimension]

	U = U_k[:,singular_indexes]
	V = VT[:,singular_indexes]*diagm(S[singular_indexes])

	return lowest_rank_TAME(A,B,U,V,β,max_iter,tol,α;profile=profile)
end

#runs TAME, but reduces down to lowest rank form first
function lowest_rank_TAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
                          U_0::Array{F,2},V_0::Array{F,2},
                          β::F, max_iter::Int,tol::F,α::F;profile=false) where {F <:AbstractFloat}

	@assert size(U_0,2) == size(V_0,2)

    dimension = minimum((A.n,B.n))

    best_triangle_count = -Inf
    best_x = zeros(size(U_0,1),size(V_0,1))
    best_index = -1


	X_k = U_0 * V_0'
	X_0 = copy(X_k)
#    X_k = copy(W)
 #   X_k_1 = copy(W)
 	rank_0 = size(U_0,2)
    i = 1
    lambda = Inf

    if profile
        experiment_profile = Dict(
            "ranks"=>[],
            "contraction_timings"=>[],
            "svd_timings"=>[],
            "matching_timings"=>[],
            "scoring_timings"=>[]
        )
    end

	U_k = copy(U_0)
	V_k = copy(V_0)

    while true

        if profile
            X_k_1,t = @timed kron_contract(A,B,U_k,V_k)
            push!(experiment_profile["contraction_timings"],t)
        else
            X_k_1 = kron_contract(A,B,U_k,V_k)
        end

        new_lambda = dot(X_k_1,X_k)

        if β != 0.0
            X_k_1 .+= β * X_k
        end

        if α != 1.0
            X_k_1 = α * X_k_1 + (1 - α) * X_0
        end

        X_k_1 ./= norm(X_k_1)

		sparse_X_k_1 = sparse(X_k_1)

        if profile
            triangles, gaped_triangles, matching_time, scoring_time = TAME_score(A,B,sparse_X_k_1;return_timings=true)
            push!(experiment_profile["matching_timings"],matching_time)
            push!(experiment_profile["scoring_timings"],scoring_time)
        else
            triangles, gaped_triangles = TAME_score(A,B,sparse_X_k_1)
        end

        println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(gaped_triangles)")

        if triangles > best_triangle_count
            best_x = copy(X_k_1)
            best_triangle_count = triangles
            best_iterate = i
        end

        println("λ: $(new_lambda)")

        if abs(new_lambda - lambda) < tol || i >= max_iter
            if profile
                return best_x, best_triangle_count, experiment_profile
            else
                return best_x, best_triangle_count
            end
        else

			#get the low rank factorization for the next one

			rank_k = size(U_k,2)
			rank_k_1 = rank_k^2 + rank_k + rank_0

			if profile
				(result,t) = @timed svds(sparse_X_k_1,nsv = rank_k_1)
				push!(experiment_profile["svd_timings"],t)
			else
				result = @timed svds(sparse_X_k_1,nsv = rank_k_1)
			end

			U,S,VT = result[1][1]
			singular_indexes = [i for i in 1:length(S) if S[i] > S[1]*eps(Float64)*dimension]

			U_k = U[:,singular_indexes]
			V_k = VT[:,singular_indexes]*diagm(S[singular_indexes])

			X_k = copy(X_k_1)
			lambda = new_lambda
			i += 1
        end

    end

end


function TAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,W::Array{F,2},
                       β::F, max_iter::Int,tol::F,α::F;profile=false) where {F <:AbstractFloat}

    dimension = minimum((A.n,B.n))

    if profile
        experiment_profile = Dict(
            "contraction_timings"=>[],
            "matching_timings"=>[],
            "hungarian_timings"=>[],
        )
    end

    best_triangle_count = -Inf
    best_index = -1
    x0 = reshape(W,A.n*B.n)
    x_k = copy(x0)

    i = 1
    lambda = Inf

    while true

        if profile
            x_k_1,t = @timed implicit_contraction(A,B,x_k)
            implicit_contraction(A,B,U,V)
            push!(experiment_profile["contraction_timings"],t)
        else
            x_k_1 = implicit_contraction(A,B,x_k)
        end

        new_lambda = dot(x_k_1,x_k)

        if β != 0.0
            x_k_1 .+= β * x_k
        end

        if α != 1.0
            x_k_1 = α * x_k_1 + (1 - α) * x0
        end

        x_k_1 ./= norm(x_k_1)

        if profile
            triangles, gaped_triangles, hungarian_time, matching_time = TAME_score(A,B,x_k_1;return_timings=true)
            push!(experiment_profile["hungarian_timings"],hungarian_time)
            push!(experiment_profile["matching_timings"],matching_time)
        else
            triangles, gaped_triangles =  TAME_score(A,B,x_k_1)
        end

        println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(gaped_triangles)")

        if triangles > best_triangle_count
            best_x = copy(x_k_1)
            best_triangle_count = triangles
            best_iterate = i
        end
        println("λ: $(new_lambda)")

        if abs(new_lambda - lambda) < tol || i >= max_iter
            if profile
                return best_x, best_triangle_count, experiment_profile
            else
                return best_x, best_triangle_count
            end
        else
            x_k = copy(x_k_1)
            lambda = new_lambda
            i += 1
        end

    end


end