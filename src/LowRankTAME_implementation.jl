

struct LowRankTAME_Return{T}
	matchScore::Union{Int,Vector{Int}}
	motifCounts::NTuple{2,Union{Int,Vector{Int}}}
	matching::Union{Dict{Int,Int},Vector{Int}}
	embedding::Tuple{Matrix{T},Matrix{T}}
	profile::Union{Nothing,Vector{Tuple{String,Dict{String,Union{Array{Float64,1},Array{Array{Float64,1},1}}}}}}
end
#=------------------------------------------------------------------------------
              Routines for searching over alpha/beta parameters
------------------------------------------------------------------------------=#
function LowRankTAME_param_search(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
						          iter::Int = 15,tol::Float64=1e-6,
						          U_0::Array{Float64,2} = ones(A.n,1),
						          V_0::Array{Float64,2} = ones(B.n,1),
						          alphas::Array{F,1}=[.5,1.0],
						          betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						          kwargs...) where {F <: AbstractFloat}
	A_motifs = size(A.indices,1)
	B_motifs = size(B.indices,1)
    max_triangle_match = min(A_motifs,B_motifs)

    total_triangles = size(A.indices,1) + size(B.indices,1)
    best_TAME_PP_tris = -1
	best_matching = Dict{Int,Int}()

	m = A.n
	n = B.n


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

	return LowRankTAME_Return(best_TAME_PP_tris,(A_motifs,B_motifs),best_matching,
	                          (best_TAME_PP_U, best_TAME_PP_V), nothing)

end

function LowRankTAME_param_search_profiled(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
                           iter::Int = 15,tol::Float64=1e-6,
						   alphas::Array{F,1}=[.5,1.0],
						   betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						   kwargs...) where {F <: AbstractFloat}
	A_motifs = size(A.indices,1)
	B_motifs = size(B.indices,1)
    max_triangle_match = min(A_motifs,B_motifs)

    total_triangles = size(A.indices,1) + size(B.indices,1)
	best_TAME_PP_tris = -1
	best_matching = Dict{Int,Int}()

	best_TAME_PP_U = ones(A.n,1)
	best_TAME_PP_V = ones(B.n,1)

	experiment_profiles = Array{Tuple{String,Dict{String,Union{Array{Float64,1},Array{Array{Float64,1},1}}}},1}(undef,0)


    for α in alphas
        for β in betas

			U, V, triangle_count,matching, experiment_profile =
				LowRankTAME_profiled(A,B,ones(A.n,1),ones(B.n,1), β,iter,tol,α;kwargs...)

			push!(experiment_profiles,("α:$(α)_β:$(β)",experiment_profile))

            if triangle_count > best_TAME_PP_tris
				best_TAME_PP_tris = triangle_count
				best_matching = matching
				best_TAME_PP_U = copy(U)
				best_TAME_PP_V = copy(V)

            end
			println("α:$(α) -- β:$(β) -- tri_match:$(best_TAME_PP_tris) -- max_tris $(max_triangle_match)")
        end

    end

	return LowRankTAME_Return(best_TAME_PP_tris,(A_motifs,B_motifs),best_matching,
							 (best_TAME_PP_U, best_TAME_PP_V), experiment_profiles)


end

#=------------------------------------------------------------------------------
             		    Spectral Relaxation Routines
------------------------------------------------------------------------------=#

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

"""------------------------------------------------------------------------------
  The low rank implementation of TAME, computes the terms use the mixed property
  generalized to the tensor case. 

  Inputs
  ------
  * A,B - (ThirdOrderSymTensor):
	The tensors to align against one another. 
  * 'U_0', 'V_0' - (Array{Float,2}):
	The low rank components of the starting iteration X = UV'. Iterates are 
	normalized before the iterations begin. 
  * β - (Float):
	The shift to use on the iterations.
  * α -(float):
	The mixing parameter for combining in the starting iterates. 
  * 'max_iter' - (Int):
	The maximum number of iterations to run. 
  * tol - (Float):
	The tolerence to solve the algorithm to. Computes the tolerance by measuring
	the absolute value of the difference between the computed eigenvalues. 
  * 'max_rank' - (Int):
	Specify the maximum rank of each of the iterates. Default makes it so that 
	only singular values small enough to be considered zero are truncated. 
  * 'update_user' - (Int):
	Specifies how frequently output messages should be printed to the user. 
	Default is -1 which means no output messages are printed, else if the value 
	is k, then the kth iterate will print out an update. 
  * 'no_matching' - (Bool):
	Specifies whether or not to run the matching and scoring portions of the 
	algorithm. Useful if only the iterates are desired. 
  * 'low_rank_matching' - (Bool):
	Specifies whether or not to run the low rank matching procedure from [1]. 
	This is useful when speed is needed, but may lead to regressions in the 
	matching performance. 

  Output
  ------
  * 'best_U', 'best_V' - (Array{Float,2})
    Returns the components to the iteration which matched the most triangles. 
  * 'best_triangle_count' - (Int)
    The maximum number of triangles matched. 
  * 'best_matching' - (Dict{Int,Int})
	The matching computed between the two graphs, maps the vertices of A to the
	vertices of B. 

  Citation
  --------
  [1] - H. Nassar, N. Veldt, S. Mohammadi, A. Grama, and D. F. Gleich, 
		“Low rank spectral network alignment,” in Proceedings of the 2018 
		World Wide Web Conference, 2018, pp. 619–628
------------------------------------------------------------------------------"""
function LowRankTAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
                          U_0::Array{F,2},V_0::Array{F,2}, β::F, max_iter::Int,tol::F,α::F;
						  max_rank::Int = minimum((A.n,B.n)),update_user::Int=-1,
						  no_matching=false,low_rank_matching=false,kwargs...) where {F <:AbstractFloat}

	@assert size(U_0,2) == size(V_0,2)

	dimension = minimum((A.n,B.n))

	best_triangle_count::Int = -1
	best_matching = Dict{Int,Int}()
    best_x = zeros(size(U_0,1),size(V_0,1))
    best_index = -1
	triangles = -1
	gaped_triangles = -1

	normalization_factor = sqrt(tr((V_0'*V_0)*(U_0'*U_0)))
	U_0 ./= sqrt(normalization_factor)
	V_0 ./= sqrt(normalization_factor)

	U_k = copy(U_0)
	V_k = copy(V_0)

	best_U::Array{F,2} = copy(U_k)
	best_V::Array{F,2} = copy(U_k)

    lambda = Inf

    for i in 1:max_iter

		A_comps, B_comps = get_kron_contract_comps(A,B,U_k,V_k)
		lam = tr((B_comps'*V_k)*(U_k'*A_comps))

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

		A_Q,A_R = qr(U_temp)
		B_Q,B_R = qr(V_temp)

		core = A_R*B_R'
		C_U,C_S,C_Vt = svd(core)
		singular_indexes= [i for i in 1:1:minimum((max_rank,length(C_S))) if C_S[i] > C_S[1]*eps(Float64)*dimension]

		U_k_1 = A_Q*C_U[:,singular_indexes]
		V_k_1 = B_Q*(C_Vt[:,singular_indexes]*diagm(C_S[singular_indexes]))

		normalization_factor = sqrt(tr((V_k_1'*V_k_1)*(U_k_1'*U_k_1)))

		U_k_1 ./= sqrt(normalization_factor)
		V_k_1 ./= sqrt(normalization_factor)

		#TODO: need to redo or remove this
		Y, Z = get_kron_contract_comps(A,B,U_k_1,V_k_1)

		lam = tr((Y'*U_k_1)*(V_k_1'*Z))


		if !no_matching
			#evaluate the matchings
			if low_rank_matching
				triangles, gaped_triangles, matching = TAME_score(A,B,U_k_1,V_k_1;kwargs...)
			else
				triangles, gaped_triangles, matching = TAME_score(A,B,U_k_1*V_k_1';kwargs...)
			end

			if triangles > best_triangle_count
				best_matching = matching
				best_triangle_count  = triangles
				best_U = copy(U_k_1)
				best_V = copy(V_k_1)
			end
		end

		if update_user != -1 && i % update_user == 0
			println("λ_$i: $(lam) -- rank:$(length(singular_indexes)) -- tris:$(triangles) -- gaped_t:$(gaped_triangles)")
		end

        if abs(lam - lambda) < tol
    		return best_U, best_V, best_triangle_count, best_matching
        end

		#get the low rank factorization for the next one
		U_k = copy(U_k_1)
		V_k = copy(V_k_1)

		lambda = lam

    end

	return best_U, best_V, best_triangle_count, best_matching
end

"""------------------------------------------------------------------------------
  The low rank implementation of TAME, computes the terms use the mixed property
  generalized to the tensor case. 

  Inputs
  ------
  * A,B - (ThirdOrderSymTensor):
	The tensors to align against one another. 
  * 'U_0', 'V_0' - (Array{Float,2}):
	The low rank components of the starting iteration X = UV'. Iterates are 
	normalized before the iterations begin. 
  * β - (Float):
	The shift to use on the iterations.
  * α -(float):
	The mixing parameter for combining in the starting iterates. 
  * 'max_iter' - (Int):
	The maximum number of iterations to run. 
  * tol - (Float):
	The tolerence to solve the algorithm to. Computes the tolerance by measuring
	the absolute value of the difference between the computed eigenvalues. 
  * 'max_rank' - (Int):
	Specify the maximum rank of each of the iterates. Default makes it so that 
	only singular values small enough to be considered zero are truncated. 
  * 'update_user' - (Int):
	Specifies how frequently output messages should be printed to the user. 
	Default is -1 which means no output messages are printed, else if the value 
	is k, then the kth iterate will print out an update. 
  * 'no_matching' - (Bool):
	Specifies whether or not to run the matching and scoring portions of the 
	algorithm. Useful if only the iterates are desired. 
  * 'low_rank_matching' - (Bool):
	Specifies whether or not to run the low rank matching procedure from [1]. 
	This is useful when speed is needed, but may lead to regressions in the 
	matching performance. 

  Output
  ------
	* 'best_U', 'best_V' - (Array{Float,2})
	  Returns the components to the iteration which matched the most triangles. 
	* 'best_triangle_count' - (Int)
	  The maximum number of triangles matched. 
	* 'best_matching' - (Dict{Int,Int})
	  The matching computed between the two graphs, maps the vertices of A to the
	  vertices of B. 
	* 'experiment_profile' - (Dict{String,Union{Array{F,1},Array{Array{F,1},1}}}):
	  The experiment profile computed, keys for the experiment data collected are
	  as follows. 
	  +'ranks' - The ranks of each iterate X_k. 
	  +'contraction_timings' - Time taken to compute each contraction. 
	  +'svd_timings' - Time taken to compute the svd.
	  +'qr_timings' - Time taken to compute the QR factorations.
	  +'matched_tris' - The number of triangles matched by each iterate. 
	  +'sing_vals' - The singular values of each iterate X_k. 
	  +'matching_times' - Time taken to solve the matchings. 
	  +'soring_timings' - Time taken to score each matching. 

  Citation
  --------
	[1] - H. Nassar, N. Veldt, S. Mohammadi, A. Grama, and D. F. Gleich, 
		  “Low rank spectral network alignment,” in Proceedings of the 2018 
		  World Wide Web Conference, 2018, pp. 619–628
------------------------------------------------------------------------------"""
function LowRankTAME_profiled(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
                          U_0::Array{F,2},V_0::Array{F,2}, β::F, max_iter::Int,tol::F,α::F;
						  max_rank::Int = minimum((A.n,B.n)),update_user::Int=-1,	
						  no_matching::Bool=false,low_rank_matching::Bool=false,kwargs...) where {F <:AbstractFloat}

	@assert size(U_0,2) == size(V_0,2)

	dimension = minimum((A.n,B.n))

	best_triangle_count::Int = -1
	best_matching = Dict{Int,Int}()
    best_x = zeros(size(U_0,1),size(V_0,1))
    best_index = -1
	triangles = -1
	gaped_triangles = -1

	normalization_factor = sqrt(tr((V_0'*V_0)*(U_0'*U_0)))
	U_0 ./= sqrt(normalization_factor)
	V_0 ./= sqrt(normalization_factor)


	experiment_profile = Dict{String,Union{Array{F,1},Array{Array{F,1},1}}}(
		"ranks"=>Array{Float64,1}(undef,0),
		"contraction_timings"=>Array{Float64,1}(undef,0),
		"svd_timings"=>Array{Float64,1}(undef,0),
		"qr_timings"=>Array{Float64,1}(undef,0),
		"matched_tris"=>Array{Float64,1}(undef,0),
		"sing_vals"=>Array{Array{Float64,1},1}(undef,0),
		"matching_timings"=>Array{Float64,1}(undef,0),
		"scoring_timings"=>Array{Float64,1}(undef,0)
	)

	U_k = copy(U_0)
	V_k = copy(V_0)

	best_U::Array{F,2} = copy(U_k)
	best_V::Array{F,2} = copy(U_k)

    lambda = Inf

    for i in 1:max_iter


		(A_comps, B_comps),t = @timed get_kron_contract_comps(A,B,U_k,V_k)
		push!(experiment_profile["contraction_timings"],t)

		lambda_k_1 = tr((B_comps'*V_k)*(U_k'*A_comps))


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

		(A_Q,A_R),t_A = @timed qr(U_temp)
		(B_Q,B_R),t_B = @timed qr(V_temp)
		push!(experiment_profile["qr_timings"],t_A + t_B)

		core = A_R*B_R'
		(C_U,C_S::Array{Float64,1},C_Vt),t = @timed svd(core)
		push!(experiment_profile["svd_timings"],t)


		singular_indexes= [i for i in 1:1:minimum((max_rank,length(C_S))) if C_S[i] > C_S[1]*eps(Float64)*dimension]
		push!(experiment_profile["sing_vals"],C_S)
		push!(experiment_profile["ranks"],float(length(singular_indexes)))

		U_k_1 = A_Q*C_U[:,singular_indexes]
		V_k_1 = B_Q*(C_Vt[:,singular_indexes]*diagm(C_S[singular_indexes]))

		normalization_factor = sqrt(tr((V_k_1'*V_k_1)*(U_k_1'*U_k_1)))

		U_k_1 ./= sqrt(normalization_factor)
		V_k_1 ./= sqrt(normalization_factor)



		if !no_matching
			#evaluate the matchings
			if low_rank_matching
				triangles, gaped_tris, matching, matching_time, scoring_time = TAME_score(A,B,U_k_1,V_k_1;return_timings=returnTimings(),kwargs...)
			else
				triangles, gaped_tris, matching, matching_time, scoring_time = TAME_score(A,B,U_k_1*V_k_1';return_timings=returnTimings(),kwargs...)
			end

			push!(experiment_profile["matched_tris"],float(triangles))
			push!(experiment_profile["matching_timings"],float(matching_time))
			push!(experiment_profile["scoring_timings"], float(scoring_time))
			
			if triangles > best_triangle_count
				best_matching = matching
				best_triangle_count  = triangles
				best_U = copy(U_k_1)
				best_V = copy(V_k_1)
			end
		end

		if update_user != -1 && i % update_user == 0
			println("λ_$i: $(lambda_k_1) -- rank:$(length(singular_indexes)) -- tris:$(triangles) -- gaped_t:$(gaped_tris)")
		end

		if abs(lambda_k_1 - lambda) < tol || i >= max_iter
			#=
			triangles,_= TAME_score(A,B,sparse(best_U*best_V');return_timings=returnTimings())
			if triangles > best_triangle_count 
				best_triangle_count = triangles
			end
			=#
			return best_U, best_V, best_triangle_count,best_matching, experiment_profile
		end
		#get the low rank factorization for the next one
		U_k = copy(U_k_1)
		V_k = copy(V_k_1)

		lambda = lambda_k_1
	end

	return best_U, best_V, best_triangle_count,best_matching, experiment_profile
end

