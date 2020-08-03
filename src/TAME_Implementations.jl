#TODO: convert two function calls into Union{COOTen,ThirdOrderSymTensor}
#=------------------------------------------------------------------------------
              Routines for searching over alpha/beta parameters
------------------------------------------------------------------------------=#
#TODO: add in kwargs for variables

function align_tensors(A,B;method::String="LambdaTAME",kwargs...)

	if method == "LambdaTAME"
		return ΛTAME_param_search(A,B;kwargs...)
	elseif method == "LowRankTAME"
		return LowRankTAME_param_search(A,B;kwargs...)
	elseif method == "TAME"
		return TAME_param_search(A,B;kwargs...)
	else
		raise(ArgumentError("method must be one of 'LambdaTAME', 'LowRankTAME', or 'TAME'."))
	end
end

function ΛTAME_param_search(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
                            iter::Int = 15,tol::Float64=1e-6,
							alphas::Array{F,1}=[.5,0],
							betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
							profile=false) where {F <: AbstractFloat}#::Union{Tuple{Int,Int,Array{F,1},Array{F,1}},Tuple{Int,Int,Array{F,1},Array{F,1},Dict{String}{Array{Number,1}}}} where {F <: AbstractFloat}

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)

	if profile
		results = Dict(
			"TAME_timings" => Array{Float64,1}(undef,length(alphas)*length(betas)),
			"Krylov Timings"=> Array{Float64,1}(undef,length(alphas)*length(betas))
		)
		exp_index = 1
	end

    U = Array{Float64,2}(undef,A.n,iter)
    V = Array{Float64,2}(undef,B.n,iter)

    best_TAME_PP_tris = -1
    best_i  = -1
    best_j = -1

    for α in alphas
        for beta in betas

			if profile
				((U,V),runtime) = @timed ΛTAME(A,B,beta,iter,tol,α)
				results["TAME_timings"][exp_index] = runtime

				#search the Krylov Subspace
				((search_tris, i, j),runtime) = @timed search_Krylov_space(A,B,U,V)
				results["Krylov Timings"][exp_index] = runtime
			    exp_index += 1
			else
				U,V = ΛTAME(A,B,beta,iter,tol,α)
				search_tris, i, j = search_Krylov_space(A,B,U,V)
			end

            if search_tris > best_TAME_PP_tris
                best_TAME_PP_tris = search_tris
                best_i = i
                best_j = j
            end
        end
    end


	if profile
	    return best_TAME_PP_tris, max_triangle_match, U[best_i,:], V[best_j,:], results
	else
		return best_TAME_PP_tris, max_triangle_match, U[best_i,:], V[best_j,:]
	end
end


function ΛTAME_param_search(A::COOTen,B::COOTen; iter::Int = 15,tol::Float64=1e-6,
							alphas::Array{F,1}=[.5,0],
							betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
							profile=false,profile_all=false) where {F <: AbstractFloat}#::Union{Tuple{Int,Int,Array{F,1},Array{F,1}},Tuple{Int,Int,Array{F,1},Array{F,1},Dict{String}{Array{Number,1}}}} where {F <: AbstractFloat}

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    best_TAME_PP_tris = -1
    best_i  = -1
    best_j = -1


	if profile
		results = Dict(
			"TAME_timings" => Array{Float64,1}(undef,length(alphas)*length(betas)),
			"Krylov Timings"=> Array{Float64,1}(undef,length(alphas)*length(betas))
		)
		exp_index = 1
	end

    U = Array{Float64,2}(undef,A.cubical_dimension,iter)
    V = Array{Float64,2}(undef,B.cubical_dimension,iter)


    for α in alphas
        for beta in betas

			if profile
				((U,V),runtime) = @timed ΛTAME(A,B,beta,iter,tol,α)
				results["TAME_timings"][exp_index] = runtime

				#search the Krylov Subspace
				((search_tris, i, j),runtime) = @timed search_Krylov_space(A,B,U,V)
				results["Krylov Timings"][exp_index] = runtime
			    exp_index += 1
			else
				U,V = ΛTAME(A,B,beta,iter,tol,α)
				search_tris, i, j = earch_Krylov_space(A,B,U,V)
			end

            if search_tris > best_TAME_PP_tris
                best_TAME_PP_tris = search_tris
                best_i = i
                best_j = j
            end
        end
    end

	if profile
	    return best_TAME_PP_tris, max_triangle_match, U[:,best_i], V[:,best_j], results
	else
		return best_TAME_PP_tris, max_triangle_match, U[:,best_i], V[:,best_j]
	end
end

#add in SparseSymmetricTensors.jl function definitions
function TAME_param_search(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
                           iter::Int = 15,tol::Float64=1e-6,
						   alphas::Array{F,1}=[.5,0],
						   betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						   profile::Bool=false,profile_aggregation="all", kwargs...) where {F <: AbstractFloat}#::Union{Tuple{Int,Int,Array{F,1}},Tuple{Int,Int,Array{F,1},Dict{String}{Array{Number,1}}}}

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)
    best_TAME_PP_tris = -1
	best_profile = Dict()

	best_TAME_PP_x = Array{Float64,2}(undef,A.n,B.n)

	if  profile && profile_aggregation == "all" || profile_aggregation == "avg"
		experiment_profiles = []
	end

    for α in alphas
        for β in betas

			if profile

				x, triangle_count, experiment_profile = TAME(A,B,ones(A.n,B.n),β,iter,tol,α;profile=profile,kwargs...)

				if profile_aggregation == "all" || profile_aggregation == "avg"
					push!(experiment_profiles,("α:$(α)_β:$(β)",experiment_profile))
				end
			else
				x, triangle_count = TAME(A,B,ones(A.n,B.n),β,iter,tol,α;profile=profile,kwargs...)
			end

            if triangle_count > best_TAME_PP_tris
                best_TAME_PP_tris = triangle_count
				if profile
					best_profile = copy(experiment_profile)
				end
				best_TAME_PP_x = copy(x)
            end

        end

    end

	if profile
		if profile_aggregation == "best"
			experiment_profiles = best_profile
		elseif profile_aggregation == "avg"
			raise(error("finish aggregation"))
		end
	end


	if profile
		return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, experiment_profiles
	else
		return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x
	end
end


function LowRankTAME_param_search(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
                           iter::Int = 15,tol::Float64=1e-6,
						   alphas::Array{F,1}=[.5,0],
						   betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						   profile::Bool=false,profile_aggregation="all", kwargs...) where {F <: AbstractFloat} #::Union{Tuple{Int,Int,Array{F,2}},Tuple{Int,Int,Array{F,2},Dict{String}{Array{Number,1}}}}

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)
    best_TAME_PP_tris = -1
	best_profile = Dict()


	best_TAME_PP_U = ones(A.n,1)
	best_TAME_PP_V = ones(B.n,1)


	if  profile && profile_aggregation == "all" || profile_aggregation == "avg"
		experiment_profiles = []
	end

    for α in alphas
        for β in betas

			if profile
				U, V, triangle_count, experiment_profile =
					LowRankTAME(A,B,ones(A.n,1),ones(B.n,1), β,iter,tol,α;profile=profile,kwargs...)

				if profile_aggregation == "all" || profile_aggregation == "avg"
					push!(experiment_profiles,("α:$(α)_β:$(β)",experiment_profile))
				end
			else
				U, V, triangle_count =
					LowRankTAME(A,B,ones(A.n,1),ones(B.n,1), β,iter,tol,α;profile=profile,kwargs...)
			end

            if triangle_count > best_TAME_PP_tris
                best_TAME_PP_tris = triangle_count
				if profile
					best_profile = copy(experiment_profile)
				end
				best_TAME_PP_U = copy(U)
				best_TAME_PP_V = copy(V)

            end

        end

    end

	if profile
		if profile_aggregation == "best"
			experiment_profiles = best_profile
		elseif profile_aggregation == "avg"
			raise(error("finish aggregation"))
		end
	end

	if profile
		return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_U, best_TAME_PP_V, experiment_profiles
	else
		return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_U, best_TAME_PP_V
	end
end

#=
function TAME_param_search(A::COOTen,B::COOTen,low_rank=true; iter::Int = 15,tol::Float64=1e-6,
						   alphas::Array{F,1}=[.5,0],
						   betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						   profile=false,profile_all=false,
						   kwargs...)::Union{Tuple{Int,Int,Array{F,2},Array{F,2}},Tuple{Int,Int,Array{F,2},Array{F,2},Dict{String}{Array{Number,1}}}} where {F <:AbstractFloat}

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)
    best_TAME_PP_tris = -1
	best_profile = Dict()

	if profile_all
		experiment_profiles = []
	end

	if low_rank
		best_TAME_PP_x = Array{Float64,2}(undef,A.n,B.n)
	else
		best_TAME_PP_U = ones(A.n,1)
		best_TAME_PP_V = ones(B.n,1)
	end

	experiment_profiles = []

    for α in alphas
        for β in betas

			if profile

				if low_rank
					U, V, triangle_count, experiment_profile =
						LowRankTAME(A,B,ones(A.n,1),ones(B.n,1), β,iter,tol,α;profile=profile,kwargs...)
				else
					x, triangle_count, experiment_profile = TAME(A,B,ones(A.n,B.n),β,iter,tol,α;profile=profile,kwargs...)
				end

				if profile_all
					push!(experiment_profiles,("α:$(α)_β:$(β)",experiment_profile))
				end
			else
				if low_rank
					U, V, triangle_count =
						LowRankTAME(A,B,ones(A.n,1),ones(B.n,1), β,iter,tol,α;kwargs...)
				else
    				x, triangle_count = TAME(A,B,ones(A.n,B.n),β,iter,tol,α;kwargs...)
				end
			end

            if triangle_count > best_TAME_PP_tris
				best_profile = copy(experiment_profile)
                best_TAME_PP_tris = triangle_count
				if low_rank
    				best_TAME_PP_U = copy(U)
					best_TAME_PP_V = copy(V)
				else
					best_TAME_PP_x = copy(x)
				end
            end

        end

    end

	if !profile_all && profile
		experiment_profiles = best_profile
	end

	if low_rank
		if profile
	    	return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_U, best_TAME_PP_V, experiment_profiles
		else
			return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_U, best_TAME_PP_V
		end
	else
		if profile
			return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, experiment_profiles
		else
			return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x
		end
	end
end
=#
#=------------------------------------------------------------------------------
             		    Spectral Relaxation Routines
------------------------------------------------------------------------------=#

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

function ΛTAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor, β::Float64,
               max_iter::Int,tol::Float64,α::Float64;update_user::Int= -1 )

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


#runs TAME, but reduces down to lowest rank form first
function LowRankTAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,W::Array{F,2},
                       β::F, max_iter::Int,tol::F,α::F;
					   max_rank::Int = minimum((A.n,B.n)),kwargs...) where {F <:AbstractFloat}

    dimension = minimum((A.n,B.n))

	(U_k,S,VT),t = @timed svd(W)
	singular_indexes = [i for i in 1:minimum((max_rank,length(C_S))) if S[i] > S[1]*eps(Float64)*dimension]

	U = U_k[:,singular_indexes]
	V = VT[:,singular_indexes]*diagm(S[singular_indexes])

	return LowRankTAME(A,B,U,V,β,max_iter,tol,α;kwargs...)
end


function LowRankTAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
                          U_0::Array{F,2},V_0::Array{F,2}, β::F, max_iter::Int,tol::F,α::F;
						  max_rank::Int = minimum((A.n,B.n)),profile=false,update_user::Int=-1) where {F <:AbstractFloat} #::Union{Tuple{Array{F,2},Array{F,2},Int},Tuple{Array{F,2},Array{F,2},Int,Dict{String}{Array{Number,1}}}}where {F <:AbstractFloat}

	@assert size(U_0,2) == size(V_0,2)

	dimension = minimum((A.n,B.n))

    best_triangle_count::Int = -1
    best_x = zeros(size(U_0,1),size(V_0,1))
    best_index = -1

	normalization_factor = sqrt(tr((V_0'*V_0)*(U_0'*U_0)))
	U_0 ./= sqrt(normalization_factor)
	V_0 ./= sqrt(normalization_factor)

     if profile
        experiment_profile::Dict{String}{Array{Number,1}} = Dict(
            "ranks"=>[],
            "contraction_timings"=>[],
            "svd_timings"=>[],
			"qr_timings"=>[],
            "matching_timings"=>[],
            "scoring_timings"=>[]
        )
    end

	U_k = copy(U_0)
	V_k = copy(V_0)

	best_U::Array{F,2} = copy(U_k)
	best_V::Array{F,2} = copy(U_k)

	i = 1
    lambda = Inf

    for _ in 1:max_iter

		if profile
			 (A_comps, B_comps),t = @timed get_kron_contract_comps(A,B,U_k,V_k)
			 push!(experiment_profile["contraction_timings"],t)
		else
			A_comps, B_comps = get_kron_contract_comps(A,B,U_k,V_k)
		end

		if α != 1.0 && β != 0.0
			U_temp = hcat(sqrt(α) * A_comps, sqrt(α * β) * U_k, sqrt(1-α) * U_0)
			V_temp = hcat(sqrt(α) * B_comps, sqrt(α * β) * V_k, sqrt(1-α) * V_0)
		elseif α != 1.0
			U_temp = hcat(sqrt(α)*A_comps, sqrt(1-α)*U_0)
			V_temp = hcat(sqrt(α)*B_comps, sqrt(1-α)*V_0)
		elseif β != 0.0
			U_temp = hcat(A_comps, sqrt(β) * U_k)
			V_temp = hcat(B_comps, sqrt(β) * V_k)
		else
			U_temp = A_comps
			V_temp = B_comps
		end


		if profile
			(A_Q,A_R),t_A = @timed qr(U_temp)
			(B_Q,B_R),t_B = @timed qr(V_temp)
			push!(experiment_profile["qr_timings"],t_A + t_B)
		else
			A_Q,A_R = qr(U_temp)
			B_Q,B_R = qr(V_temp)
		end

		core = A_R*B_R'
		if profile
			(C_U,C_S,C_Vt),t = @timed svd(core)
			push!(experiment_profile["svd_timings"],t)
		else
			C_U,C_S,C_Vt = svd(core)
		end


		singular_indexes= [i for i in 1:1:minimum((max_rank,length(C_S))) if C_S[i] > C_S[1]*eps(Float64)*dimension]
		if profile
			push!(experiment_profile["ranks"],length(singular_indexes))
		end

		U_k_1 = A_Q*C_U[:,singular_indexes]
		V_k_1 = B_Q*(C_Vt[:,singular_indexes]*diagm(C_S[singular_indexes]))

		normalization_factor = sqrt(tr((V_k_1'*V_k_1)*(U_k_1'*U_k_1)))

		U_k_1 ./= sqrt(normalization_factor)
		V_k_1 ./= sqrt(normalization_factor)

		#TODO: need to redo or remove this
		Y, Z = get_kron_contract_comps(A,B,U_k_1,V_k_1)

		lam = tr((Y'*U_k_1)*(V_k_1'*Z))


		#evaluate the matchings
		sparse_X_k_1 = sparse(U_k_1*V_k_1')

		if profile
			triangles, gaped_triangles, matching_time, scoring_time = TAME_score(A,B,sparse_X_k_1;return_timings=true)
			push!(experiment_profile["matching_timings"],matching_time)
			push!(experiment_profile["scoring_timings"], scoring_time)
		else
			triangles, gaped_triangles = TAME_score(A,B,sparse_X_k_1)
		end

		if triangles > best_triangle_count
			best_triangle_count  = triangles
			best_U = copy(U_k_1)
			best_V = copy(U_k_1)
		end

		if update_user != -1 && i % update_user == 0
			println("λ_$i: $(lam) -- rank:$(length(singular_indexes)) -- tris:$(triangles) -- gaped_t:$(gaped_triangles)")
		end

        if abs(lam - lambda) < tol || i >= max_iter
    		if profile
    			return best_U, best_V, best_triangle_count, experiment_profile
			else
				return best_U, best_V, best_triangle_count
			end
        end

		#get the low rank factorization for the next one
		U_k = copy(U_k_1)
		V_k = copy(V_k_1)

		lambda = lam
		i += 1

    end

end



function TAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,W::Array{F,2},
                       β::F, max_iter::Int,tol::F,α::F;profile=false,update_user::Int=-1) where {F <:AbstractFloat}#::Union{Tuple{Array{F,2},Int},Tuple{Array{F,2},Int,Dict{String}{Array{Number,1}}}} where {F <:AbstractFloat}

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
    best_triangle_count = -1
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

		if update_user != -1 && i % update_user == 0
        	println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(gaped_triangles)")
			println("λ: $(new_lambda)")
		end

        if triangles > best_triangle_count
            best_x = copy(x_k_1)
            best_triangle_count = triangles
            best_iterate = i
        end


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