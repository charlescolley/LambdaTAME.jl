
#=------------------------------------------------------------------------------
              Routines for searching over alpha/beta parameters
------------------------------------------------------------------------------=#
#TODO: add in kwargs for variables

function align_tensors(A,B;method::String="LambdaTAME",kwargs...)

	if method == "LambdaTAME"
		return ΛTAME_param_search(A,B;kwargs...)
	elseif method == "LowRankTAME"
		return TAME_param_search(A,B,true;kwargs...)
	elseif method == "TAME"
		return TAME_param_search(A,B,false;kwargs...)
	else
		raise(ArgumentError("method must be one of 'LambdaTAME', 'LowRankTAME', or 'TAME'."))
	end
end

function ΛTAME_param_search(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
                            iter::Int = 15,tol::Float64=1e-6,
							alphas::Array{Float64,1}=[.5,0],
							betas::Array{Float64,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
							kwargs...)

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


function ΛTAME_param_search(A::COOTen,B::COOTen; iter::Int = 15,tol::Float64=1e-6,
							alphas::Array{Float64,1}=[.5,0],
							betas::Array{Float64,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001])

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
function TAME_param_search(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,low_rank=true;
                           iter::Int = 15,tol::Float64=1e-6,
						   alphas::Array{Float64,1}=[.5,0],
						   betas::Array{Float64,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						   kwargs...)

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)

    Timings = Array{Float64,1}(undef,length(alphas)*length(betas))
    best_TAME_PP_tris = -1
    best_i  = -1
    best_j = -1
    exp_index = 1

    for α in alphas
        for beta in betas
            if low_rank
                (_, triangle_count),runtime =
                    @timed LowRankTAME(A,B,ones(A.n,1),ones(B.n,1),
                                            beta,iter,tol,α;kwargs...)
            else
                (_, triangle_count),runtime =
                    @timed TAME(A,B,ones(A.n,B.n),beta,iter,tol,α;kwargs...)

            end

            Timings[exp_index] = runtime

            if triangle_count > best_TAME_PP_tris
                best_TAME_PP_tris = triangle_count
            end
            exp_index += 1
        end

    end

    avg_Timings = sum(Timings)/length(Timings)

    return best_TAME_PP_tris,max_triangle_match,total_triangles, avg_Timings

end

function TAME_param_search(A::COOTen,B::COOTen,low_rank=true; iter::Int = 15,tol::Float64=1e-6,
						   alphas::Array{Float64,1}=[.5,0],
						   betas::Array{Float64,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						   kwargs...)

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)

    Timings = Array{Float64,1}(undef,length(alphas)*length(betas))

    best_TAME_PP_tris = -1
    best_i  = -1
    best_j = -1
    exp_index = 1

    for α in alphas
        for beta in betas
            if low_rank
                (_, triangle_count),runtime =
                    @timed LowRankTAME(A,B,ones(A.n,1),ones(B.n,1),
                                            beta,iter,tol,α;kwargs...)
            else
                (_, triangle_count),runtime =
                    @timed TAME(A,B,ones(A.n,B.n),beta,iter,tol,α;kwargs...)

            end

            Timings[exp_index] = runtime

            if triangle_count > best_TAME_PP_tris
                best_TAME_PP_tris = triangle_count
            end
            exp_index += 1
        end

    end

    avg_Timings = sum(Timings)/length(Timings)

    return best_TAME_PP_tris,max_triangle_match,total_triangles, avg_Timings

end
#=------------------------------------------------------------------------------
             		    Spectral Relaxation Routines
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
function LowRankTAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,W::Array{F,2},
                       β::F, max_iter::Int,tol::F,α::F;profile=false) where {F <:AbstractFloat}

    dimension = minimum((A.n,B.n))

	(U_k,S,VT),t = @timed svd(W)
	singular_indexes = [i for i in 1:length(S) if S[i] > S[1]*eps(Float64)*dimension]

	U = U_k[:,singular_indexes]
	V = VT[:,singular_indexes]*diagm(S[singular_indexes])

	return lowest_rank_TAME(A,B,U,V,β,max_iter,tol,α;profile=profile)
end

function LowRankTAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
                          U_0::Array{F,2},V_0::Array{F,2},
                          β::F, max_iter::Int,tol::F,α::F;profile=false) where {F <:AbstractFloat}

	@assert size(U_0,2) == size(V_0,2)

    dimension = minimum((A.n,B.n))

    best_triangle_count = -Inf
    best_x = zeros(size(U_0,1),size(V_0,1))
    best_index = -1

	X_k = U_0 * V_0'
	X_0 = copy(X_k)

	U_k = copy(U_0)
	V_k = copy(V_0)

	U_k ./= norm(U_k)
	V_k ./= norm(V_k)


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

    while true

        if profile
            #X_k_1,t = @timed kron_contract(A,B,U_k,V_k)
		    (A_comps, B_comps),t = @timed get_kron_contract_comps(A,B,U_k,V_k)
            push!(experiment_profile["contraction_timings"],t)
        else
            A_comps, B_comps = get_kron_contract_comps(A,B,U_k,V_k)
        end

		U_temp = hcat(sqrt(α)*A_comps,sqrt(α*β)*U_k, sqrt(1-α)*U_0)
		V_temp = hcat(sqrt(α)*B_comps,sqrt(α*β)*V_k, sqrt(1-α)*V_0)

	    if profile
            #X_k_1,t = @timed kron_contract(A,B,U_k,V_k)
			(result_A),t_A = @timed svd(U_temp)
			#println(typeof(result_A))
			A_U,A_S,A_Vt = result_A.U,result_A.S,result_A.Vt
			(result_B),t_B = @timed svd(V_temp)
			B_U,B_S,B_Vt = result_B.U,result_B.S,result_B.Vt
            push!(experiment_profile["svd_timings"],t_A + t_B)
        else
			A_U,A_S,A_Vt = svd(U_temp)
			B_U,B_S,B_Vt = svd(V_temp)
		end
		singular_indexes_A = [i for i in 1:length(A_S) if A_S[i] > A_S[1]*eps(Float64)*dimension]
		singular_indexes_B = [i for i in 1:length(B_S) if B_S[i] > B_S[1]*eps(Float64)*dimension]

		U_k_1 = A_U[:,singular_indexes_A]*diagm(A_S[singular_indexes_A])
		V_k_1 = (B_U[:,singular_indexes_B]*diagm(B_S[singular_indexes_B]))*(B_Vt[:,singular_indexes_B]'*A_Vt[:,singular_indexes_A])


		new_lambda = dot(V_k_1'*V_k, U_k_1'*U_k)

		U_k_1 ./= norm(U_k_1)
		V_k_1 ./= norm(V_k_1)

		X_k_1 = U_k_1*V_k_1'
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
			U_k = copy(U_k_1)
			V_k = copy(V_k_1)

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
            "scoring_timings"=>[],
			"matched/gaped triangles"=>[],
			"ranks"=>[]
        )
    end

	best_x = Array{Float64,1}(undef,A.n*B.n)
    best_triangle_count = -Inf
    best_index = -1
    x0 = reshape(W,A.n*B.n)
	x0 ./=norm(x0)
    x_k = copy(x0)

    i = 1
    lambda = Inf

    while true

        if profile
            x_k_1,t = @timed implicit_contraction(A,B,x_k)
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
			push!(experiment_profile["ranks"],rank(reshape(x_k_1,(A.n,B.n))))
		end

		sparse_X_k_1 = sparse(reshape(x_k_1,A.n,B.n))

        if profile
            triangles, gaped_triangles, bipartite_matching_time, scoring_time = TAME_score(A,B,sparse_X_k_1;return_timings=true)
            push!(experiment_profile["matching_timings"],bipartite_matching_time)
            push!(experiment_profile["scoring_timings"],scoring_time)
			push!(experiment_profile["matched/gaped triangles"],(triangles, gaped_triangles))
        else
            triangles, gaped_triangles =  TAME_score(A,B,sparse_X_k_1)
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