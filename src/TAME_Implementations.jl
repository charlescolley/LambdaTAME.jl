#TODO: convert two function calls into Union{COOTen,ThirdOrderSymTensor}
#=------------------------------------------------------------------------------
              Routines for searching over alpha/beta parameters
------------------------------------------------------------------------------=#
"""------------------------------------------------------------------------------
  This function aligns graphs using their tensor representations. These routines
  call the param_search functions for the associated method used.

  Inputs
  ------
  * A, B - (ThirdOrderSymTensor):
	Two third order tensors representing the presence of triangles within the 
	network. A must be larger than B, else the routines will be called with the
	parameters swapped. 
  * method - (String):
	The choice of method used to align the methods. Options include 'LambdaTAME',
	'LowRankTAME', and 'TAME'. 
  
  Outputs
  -------
	* 'best_TAME_PP_tris' - (Int):
		The largest number of triangles matched over all iterations.
	* 'max_triangle_match' - (Int):
	  The maximum number of triangles matchable. This is the minimum between the 
	  number of triangles in graphs A and B. 
	* 'best_TAME_PP_x' - (Array{Float,2}):
	  The best iterate found over all the alphas and betas specified by the user. 
	  when 'LambdaTAME' and 'LowRankTAME' are called, this is replaced by the U
	  and V components of the best iterate. 

-----------------------------------------------------------------------------"""
function align_tensors(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor;
					   method::String="LambdaTAME",no_matching=false,kwargs...)

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

"""------------------------------------------------------------------------------
  This function aligns graphs using their tensor representations and returns the
  profiled version of the algorithms. These routines call the 'param_search'
  functions for the associated method used.

  Inputs
  ------
  * A, B - (ThirdOrderSymTensor):
	Two third order tensors representing the presence of triangles within the 
	network. A must be larger than B, else the routines will be called with the
	parameters swapped. 
  * method - (String):
	The choice of method used to align the methods. Options include 'LambdaTAME',
	'LowRankTAME', and 'TAME'. 
  * 'no_matching' - (Bool):
	Will not run the matching routines when True. This is useful when studying 
	the ranks of the iterates. Any counts which may rely on the matchings are 
	replaced by -1. 
  Outputs
  -------
  * 'best_TAME_PP_tris' - (Int):
    The largest number of triangles matched over all iterations.
  * 'max_triangle_match' - (Int):
	The maximum number of triangles matchable. This is the minimum between the 
	number of triangles in graphs A and B. 
  * 'best_TAME_PP_x' - (Array{Float,2}):
	The best iterate found over all the alphas and betas specified by the user. 
	when 'LambdaTAME' and 'LowRankTAME' are called, this is replaced by the U
	and V components of the best iterate. 
  * 'best_matching' - (Dict{Int,Int}):
	The best matching found, maps from A to B. When the method flips A and B, 
	the dictionary is also flipped when returned. 
  * profile - (Dict):
	A dictionary storing the profiling results from each of the methods. Please 
	see the '_profiled' versions of the code to see what is returned by each 
	function.
------------------------------------------------------------------------------"""
function align_tensors_profiled(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor;
					            method::String="LambdaTAME",no_matching=false,kwargs...)

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


function ΛTAME_param_search(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
                            iter::Int = 15,tol::Float64=1e-6,
							alphas::Array{F,1}=[.5,1.0],
							betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001]) where {F <: AbstractFloat}

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)

    U = Array{Float64,2}(undef,A.n,iter)
    V = Array{Float64,2}(undef,B.n,iter)

    best_TAME_PP_tris = -1
    best_i  = -1
	best_j = -1
	best_matching = Dict{Int,Int}()

    for α in alphas
        for beta in betas

			U,V = ΛTAME(A,B,beta,iter,tol,α)
			search_tris, i, j, matching = search_Krylov_space(A,B,U,V)
			println("α:$(α) -- β:$(beta) finished -- tri_match:$search_tris -- max_tris $(max_triangle_match) -- best tri_match:$best_TAME_PP_tris")
            if search_tris > best_TAME_PP_tris
                best_TAME_PP_tris = search_tris
                best_i = i
				best_j = j
				best_matching = matching
            end
        end
    end
	println("best i:$best_i -- best j:$best_j")
	return best_TAME_PP_tris, max_triangle_match, U[best_i,:], V[best_j,:], best_matching
end


function ΛTAME_param_search_profiled(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor; 
	                                 iter::Int = 15,tol::Float64=1e-6, alphas::Array{F,1}=[.5,1.0],
							         betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001]) where {F <: AbstractFloat}

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    best_TAME_PP_tris = -1
    best_i  = -1
	best_j = -1
	best_matching = Dict{Int,Int}()

	m = A.n
	n = B.n



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
function TAME_param_search(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
                           iter::Int = 15,tol::Float64=1e-6, alphas::Array{F,1}=[.5,1.0],
						   betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						   kwargs...) where {F <: AbstractFloat}

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)
	best_TAME_PP_tris::Int = -1
	best_matching = Dict{Int,Int}()

	m = A.n
	n = B.n

	best_TAME_PP_x = Array{Float64,2}(undef,m,n)


    for α in alphas
        for β in betas

			x, triangle_count, matching = TAME(A,B,β,iter,tol,α;W = ones(A.n,B.n),kwargs...)

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

function TAME_param_search_profiled(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
                                    iter::Int = 15,tol::Float64=1e-6, alphas::Array{F,1}=[.5,1.0],
						            betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						            profile::Bool=false,profile_aggregation="all",
						            kwargs...) where {F <: AbstractFloat}
    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)
	best_TAME_PP_tris = -1
	best_matching = Dict{Int,Int}()

	m = A.n
	n = B.n

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

function LowRankTAME_param_search(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
						          iter::Int = 15,tol::Float64=1e-6,
						          U_0::Array{Float64,2} = ones(A.n,1),
						          V_0::Array{Float64,2} = ones(B.n,1),
						          alphas::Array{F,1}=[.5,1.0],
						          betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						          kwargs...) where {F <: AbstractFloat}
    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
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

	return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_U, best_TAME_PP_V, best_matching

end

function LowRankTAME_param_search_profiled(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
                           iter::Int = 15,tol::Float64=1e-6,
						   alphas::Array{F,1}=[.5,1.0],
						   betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						   kwargs...) where {F <: AbstractFloat}
    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)
	best_TAME_PP_tris = -1
	best_matching = Dict{Int,Int}()

	best_TAME_PP_U = ones(A.n,1)
	best_TAME_PP_V = ones(B.n,1)

	experiment_profiles = []


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

	return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_U, best_TAME_PP_V,best_matching, experiment_profiles

end
#=------------------------------------------------------------------------------
             		    Spectral Relaxation Routines
------------------------------------------------------------------------------=#

"""------------------------------------------------------------------------------
  The LambdaTAME method. Method starts with a uniform prior and computes the 
  power method for each of the tensors passed it. These iterates are stored and
  the best rank one matching is picked out of all the quadratic pairs of vectors
  computed. This portion of the algorith computes the contractions, the full 
  procedure can be found in the associated '_param_search' algorithms. 

  Inputs
  ------
  * A,B - (ThirdOrderSymTensor):
	The tensors to align against one another. 
  * β - (Float):
	The shift to use on the iterations.
  * α -(float):
	The mixing parameter for combining in the starting iterates. 
  * 'max_iter' - (Int):
	The maximum number of iterations to run. 
  * tol - (Float):
	The tolerence to solve the algorithm to. Computes the tolerance by measuring
	the absolute value of the difference between the computed eigenvalues. 
  * 'update_user' - (Int):
	Specifies how frequently output messages should be printed to the user. 
	Default is -1 which means no output messages are printed, else if the value 
	is k, then the kth iterate will print out an update. 
  Output
  ------
  * U, V - (Array{Float,2})
    Returns the computed contractions started with a uniform prior.
------------------------------------------------------------------------------"""
function ΛTAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor, β::Float64,
               max_iter::Int,tol::Float64,α::Float64;update_user=-1)

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
						  no_matching=false,low_rank_matching=false) where {F <:AbstractFloat}

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
				triangles, gaped_triangles,matching = TAME_score(A,B,U_k_1,V_k_1)
			else
				triangles, gaped_triangles,matching = TAME_score(A,B,U_k_1*V_k_1')
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
						  no_matching::Bool=false,low_rank_matching::Bool=false) where {F <:AbstractFloat}

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
				triangles, gaped_tris, matching, matching_time, scoring_time = TAME_score(A,B,U_k_1,V_k_1;return_timings=true)
			else
				triangles, gaped_tris, matching, matching_time, scoring_time = TAME_score(A,B,U_k_1*V_k_1';return_timings=true)
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
			triangles,_= TAME_score(A,B,sparse(best_U*best_V');return_timings=false)
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

function setup_tame_data(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor)
	return _index_triangles_nodesym(A.n,A.indices), _index_triangles_nodesym(B.n,B.indices)
end

function _index_triangles_nodesym(n,Tris::Array{Int,2})

	Ti = [ Vector{Tuple{Int,Int}}(undef, 0) for i in 1:n ]
	for (ti,tj,tk) in eachrow(Tris)
		push!(Ti[ti], (tj,tk))
		push!(Ti[tj], (ti,tk))
		push!(Ti[tk], (ti,tj))
	end
	sort!.(Ti)
	return Ti
end

"""------------------------------------------------------------------------------
  An implementation of TAME, for our experiments we use the original C++ code, 
  but we include these routines for any additional experimentation desired. 

  Inputs
  ------
  * A,B - (ThirdOrderSymTensor):
	The tensors to align against one another. 
  * W - (Array{Float,2}):
	The starting iteration, Iterate is normalized before the iterations begin. 
  * β - (Float):
	The shift to use on the iterations.
  * α -(float):
	The mixing parameter for combining in the starting iterates. 
  * 'max_iter' - (Int):
	The maximum number of iterations to run. 
  * tol - (Float):
	The tolerence to solve the algorithm to. Computes the tolerance by measuring
	the absolute value of the difference between the computed eigenvalues. 
  * 'update_user' - (Int):
	Specifies how frequently output messages should be printed to the user. 
	Default is -1 which means no output messages are printed, else if the value 
	is k, then the kth iterate will print out an update. 
  * 'no_matching' - (Bool):
	Specifies whether or not to run the matching and scoring portions of the 
	algorithm. Useful if only the iterates are desired. 

  Output
  ------
	* 'best_x'- (Array{Float,2})
	  Returns the components to the iteration which matched the most triangles. 
	  Reshapes the iterate x into a matrix. 
	* 'best_triangle_count' - (Int)
	  The maximum number of triangles matched. 
	* 'best_matching' - (Dict{Int,Int})
	  The matching computed between the two graphs, maps the vertices of A to the
	  vertices of B. 
------------------------------------------------------------------------------"""
function TAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,β::F, max_iter::Int,
			  tol::F,α::F;update_user::Int=-1,W::Array{F,2}=ones(A.n,B.n),
			  no_matching=false,) where {F <:AbstractFloat}

    dimension = minimum((A.n,B.n))

	A_Ti, B_Ti = setup_tame_data(A,B)

	best_x = Array{Float64,2}(undef,A.n,B.n)
    best_triangle_count = -1
	best_index = -1
	best_matching = Dict{Int,Int}()

    x0 = reshape(W,A.n*B.n)
	x0 ./=norm(x0)
    x_k = copy(x0)

    i = 1
    lambda = Inf

    while true

	    x_k_1 = impTTVnodesym(A.n, B.n, x_k, A_Ti, B_Ti)
		#x_k_1 = implicit_contraction(A,B,x_k)

		#println("norm diff is:",norm(x_k_1_test - x_k_1)/norm(x_k_1_test))
        new_lambda = dot(x_k_1,x_k)

        if β != 0.0
            x_k_1 .+= β * x_k
        end

        if α != 1.0
            x_k_1 = α * x_k_1 + (1 - α) * x0
        end

		x_k_1 ./= norm(x_k_1)
		
		if !no_matching

			X = reshape(x_k_1,A.n,B.n)
			triangles, gaped_triangles, matching =  TAME_score(A,B,X)

			if update_user != -1 && i % update_user == 0
				println("finished iterate $(i):tris:$triangles -- gaped_t:$gaped_triangles")
			end

			if triangles > best_triangle_count
				best_matching = matching
				best_x = copy(x_k_1)
				best_triangle_count = triangles
				best_iterate = i
			end

		end

		if update_user != -1 && i % update_user == 0
			println("λ: $(new_lambda)")
		end


        if abs(new_lambda - lambda) < tol || i >= max_iter
  			return reshape(best_x,A.n,B.n), best_triangle_count, best_matching
        else
            x_k = copy(x_k_1)
            lambda = new_lambda
            i += 1
        end

    end


end

"""------------------------------------------------------------------------------
  The low rank implementation of TAME, computes the terms use the mixed property
  generalized to the tensor case. 

  Inputs
  ------
  * A,B - (ThirdOrderSymTensor):
	The tensors to align against one another. 
  * W - (Array{Float,2}):
	The starting iteration, Iterate is normalized before the iterations begin. 
  * β - (Float):
	The shift to use on the iterations.
  * α -(float):
	The mixing parameter for combining in the starting iterates. 
  * 'max_iter' - (Int):
	The maximum number of iterations to run. 
  * tol - (Float):
	The tolerence to solve the algorithm to. Computes the tolerance by measuring
	the absolute value of the difference between the computed eigenvalues. 
  * 'update_user' - (Int):
	Specifies how frequently output messages should be printed to the user. 
	Default is -1 which means no output messages are printed, else if the value 
	is k, then the kth iterate will print out an update. 
  * 'no_matching' - (Bool):
	Specifies whether or not to run the matching and scoring portions of the 
	algorithm. Useful if only the iterates or profiling is desired. 

  Output
  ------
    * 'best_x'- (Array{Float,2})
      Returns the components to the iteration which matched the most triangles. 
      Reshapes the iterate x into a matrix.  
	* 'best_triangle_count' - (Int)
	  The maximum number of triangles matched. 
	* 'best_matching' - (Dict{Int,Int})
	  The matching computed between the two graphs, maps the vertices of A to the
	  vertices of B. 
	* 'experiment_profile' - (Dict{String,Union{Array{F,1},Array{Array{F,1}}}}):
	  The experiment profile computed, keys for the experiment data collected are
	  as follows. 
	  +'contraction_timings' - Time taken to compute each contraction. 
	  +'matched_tris' - The number of triangles matched by each iterate. 
	  +'gaped triangles' - The number of unmatched triangles, equal to the 
	                       maximum - matched triangles.
	  +'sing_vals' - The singular values of each iterate X_k. 
	  +'ranks' - The ranks of each iterate X_k. 
	  +'matching_timings' - Time taken to solve the matchings. 
	  +'soring_timings' - Time taken to score each matching. 

------------------------------------------------------------------------------"""
function TAME_profiled(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor, β::F, max_iter::Int,
	                   tol::F,α::F;update_user::Int=-1,W::Array{F,2} = ones(m,n),
					   no_matching::Bool=false) where {F <:AbstractFloat}

    dimension = minimum((A.n,B.n))

    experiment_profile = Dict{String,Union{Array{F,1},Array{Array{F,1}}}}(
		"contraction_timings"=>Array{F,1}(undef,0),
		"matching_timings"=>Array{F,1}(undef,0),
		"scoring_timings"=>Array{F,1}(undef,0),
		"matched triangles"=>Array{F,1}(undef,0),
		"gaped triangles"=>Array{F,1}(undef,0),
		"sing_vals"=>Array{Array{F,1},1}(undef,0),
		"ranks"=>Array{F,1}(undef,0)
	)


	A_Ti, B_Ti = setup_tame_data(A,B)

	best_x = Array{Float64,2}(undef,A.n,B.n)
    best_triangle_count::Int = -1
	best_index = -1
	best_matching = Dict{Int,Int}()

    x0 = reshape(W,A.n*B.n)
	x0 ./=norm(x0)
    x_k = copy(x0)

    i = 1
    lambda = Inf

    while true

		x_k_1,t = @timed impTTVnodesym(A.n, B.n, x_k, A_Ti, B_Ti)
		push!(experiment_profile["contraction_timings"],t)

        new_lambda = dot(x_k_1,x_k)

        if β != 0.0
            x_k_1 .+= β * x_k
        end

        if α != 1.0
            x_k_1 = α * x_k_1 + (1 - α) * x0
        end

        x_k_1 ./= norm(x_k_1)

		S = svdvals(reshape(x_k_1,(A.n,B.n)))
		
		rank = 0.0
		for i in 1:length(S)
			if S[i] > S[1]*eps(Float64)*dimension
				rank = rank + 1
			end
		end

		
		push!(experiment_profile["sing_vals"],S)
		push!(experiment_profile["ranks"],rank)

		if !no_matching

			triangles, gaped_triangles, matching, matching_time, scoring_time = TAME_score(A,B,reshape(x_k_1,A.n,B.n);return_timings=true)
			
			push!(experiment_profile["matching_timings"],matching_time)
			push!(experiment_profile["scoring_timings"],scoring_time)
			push!(experiment_profile["matched triangles"],float(triangles))
			push!(experiment_profile["gaped triangles"],float(gaped_triangles))


			if update_user != -1 && i % update_user == 0
				println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(gaped_triangles)")
			end

			if triangles > best_triangle_count
				best_matching = matching
				best_x = copy(x_k_1)
				best_triangle_count = triangles
				best_iterate = i
			end
		end

		if update_user != -1 && i % update_user == 0
			println("λ_$i: $(new_lambda)")
		end

        if abs(new_lambda - lambda) < tol || i >= max_iter
   			return reshape(best_x,A.n,B.n), best_triangle_count, best_matching, experiment_profile
        else
            x_k = copy(x_k_1)
            lambda = new_lambda
            i += 1
        end

    end


end