

abstract type MatchingMethod end
struct ΛTAME_GramMatching <: MatchingMethod end
struct ΛTAME_rankOneMatching <: MatchingMethod end


struct ΛTAME_Return{T} <: returnType
	matchScore::Union{Int,Vector{Int}}
	motifCounts::NTuple{2,Union{Int,Vector{Int}}}
	matching::Union{Dict{Int,Int},Vector{Int}}
	rank1Matchingindices::Union{Nothing,Tuple{Int,Int}}
	embedding::Tuple{Matrix{T},Matrix{T}}
	profile::Union{Nothing,Dict{String,Array{Float64,1}}}
end

struct ΛTAME_MultiMotif_Return{T} <: returnType
#    motifDistributions::NTuple(2,Vector{T}) # ttv({A,B},1,modes=k-1)
	aggregateMotifScore::Int
	matchScore::Union{Int,Vector{Int}}
	motifCounts::NTuple{2,Union{Int,Vector{Int}}}
	matching::Union{Dict{Int,Int},Vector{Int}}
	rank1Matchingindices::Union{Nothing,Tuple{Int,Int}}
	embedding::Tuple{Matrix{T},Matrix{T}}
	profile::Union{Nothing,Dict{String,Array{Float64,1}}}
end

#=------------------------------------------------------------------------------
              Routines for searching over alpha/beta parameters
------------------------------------------------------------------------------=#


function ΛTAME_param_search(A,B;#duck type tensor inputs
                            iter::Int = 15,tol::Float64=1e-6,
							alphas::Array{F,1}=[.5,1.0],
							betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
							matchingMethod::MatchingMethod=ΛTAME_rankOneMatching(),kwargs...) where {F <: AbstractFloat}
	if typeof(A) === ThirdOrderSymTensor 
		motifA = size(A.indices,1)
		motifB = size(B.indices,1)
	else
		motifA = size(A.indices,2)
		motifB = size(B.indices,2)

	end
	max_triangle_match = min(motifA ,motifB)
    #max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)

    U = Array{Float64,2}(undef,A.n,iter)
    V = Array{Float64,2}(undef,B.n,iter)
	best_U = Array{Float64,2}(undef,A.n,iter)
	best_V = Array{Float64,2}(undef,B.n,iter)

    best_TAME_PP_tris = -1
    best_i  = -1
	best_j = -1
	best_matching = Dict{Int,Int}()

    for α in alphas
        for beta in betas

			U,V = ΛTAME(A,B,beta,iter,tol,α)
			if typeof(matchingMethod) === ΛTAME_rankOneMatching
				search_tris, i, j, matching = search_Krylov_space(A,B,U,V)
			
			elseif typeof(matchingMethod) === ΛTAME_GramMatching
				search_tris, _,matching = TAME_score(A,B,U*V';kwargs...)
				i = -1
				j = -1
			else 
				throw(TypeError("typeof MatchingMethod must be either ΛTAME_rankOneMatching or ΛTAME_GramMatching"))
			end
			println("α:$(α) -- β:$(beta) finished -- tri_match:$search_tris -- max_tris $(max_triangle_match) -- best tri_match:$best_TAME_PP_tris")
            if search_tris > best_TAME_PP_tris
                best_TAME_PP_tris = search_tris
                best_i = i
				best_j = j
				best_U = U
				best_V = V
				best_matching = matching
            end
        end
    end
	println("best i:$best_i -- best j:$best_j")
	if typeof(matchingMethod) === ΛTAME_rankOneMatching
		return ΛTAME_Return(best_TAME_PP_tris,(motifA,motifB),
						    best_matching,(best_i, best_j),(best_U, best_V),nothing)
	else
		return ΛTAME_Return(best_TAME_PP_tris,(motifA,motifB),
							best_matching,nothing,(best_U, best_V),nothing)
	end

end





function ΛTAME_param_search(A_tensors::Array{SymTensorUnweighted{S},1},B_tensors::Array{SymTensorUnweighted{S},1};
							iter::Int = 15,tol::Float64=1e-6,
							alphas::Array{F,1}=[.5,1.0],
							betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
							matchingMethod::MatchingMethod=ΛTAME_rankOneMatching(),kwargs...) where {F <: AbstractFloat, S <: Motif}

	motifCountA = [size(A.indices,2) for A in A_tensors]
	motifCountB = [size(B.indices,2) for B in B_tensors]
	max_motif_match = [min(A_motifCount,B_motifCount) for (A_motifCount,B_motifCount) in zip(motifCountA,motifCountB)]
	total_triangles = [size(A.indices,2) + size(B.indices,2) for (A,B) in zip(A_tensors,B_tensors)]
	#total_triangles = size(A.indices,1) + size(B.indices,1)

	m = maximum([A.n for A in A_tensors])
	n = maximum([B.n for B in B_tensors])

	U = Array{Float64,2}(undef,m,iter)
	V = Array{Float64,2}(undef,n,iter)
	best_U = Array{Float64,2}(undef,m,iter)
	best_V = Array{Float64,2}(undef,n,iter)

	best_matching_score = -1
	best_i  = -1
	best_j = -1
	best_matched_motifs::Array{Int,1} = []
	best_matching = Dict{Int,Int}()

	for α in alphas
		for beta in betas

			U,V = ΛTAME(A_tensors,B_tensors,beta,iter,tol,α)

			if typeof(matchingMethod) === ΛTAME_rankOneMatching
				matching_score, matched_motifs, i, j, matching = search_Krylov_space(A_tensors,B_tensors,U,V)
			elseif typeof(matchingMethod) === ΛTAME_GramMatching
				matching_score, matched_motifs, matching = TAME_score(A_tensors,B_tensors,U*V';kwargs...)
				i = -1
				j = -1
			else 
				throw(TypeError("typeof MatchingMethod must be either ΛTAME_rankOneMatching or ΛTAME_GramMatching"))
			end

			if matching_score > best_matching_score
				best_matching_score = matching_score
				best_matched_motifs = matched_motifs
				best_i = i
				best_j = j
				best_U = U
				best_V = V
				best_matching = matching
			end
			println("α:$(α) -- β:$(beta) finished -- motif_match:$matched_motifs -- max_motifs:$max_motif_match -- matching_score:$matching_score -- best score:$best_matching_score")
			
		end
	end
	
	println("best i:$best_i -- best j:$best_j")
	if typeof(matchingMethod) === ΛTAME_rankOneMatching
		return ΛTAME_MultiMotif_Return(best_matching_score,best_matched_motifs, (motifCountA,motifCountB),
					best_matching,(best_i, best_j),(best_U, best_V),nothing)
	else
		return ΛTAME_MultiMotif_Return(best_matching_score,best_matched_motifs, (motifCountA,motifCountB),
					best_matching,nothing,(best_U, best_V),nothing)
	end

end

function ΛTAME_param_search_profiled(A,B; 
	                                 iter::Int = 15,tol::Float64=1e-6, alphas::Array{F,1}=[.5,1.0],
									 betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
									 matchingMethod::MatchingMethod=ΛTAME_rankOneMatching(),kwargs...) where {F <: AbstractFloat}

	if typeof(A) === ThirdOrderSymTensor 
		motifA = size(A.indices,1)
		motifB = size(B.indices,1)
	else
		motifA = size(A.indices,2)
		motifB = size(B.indices,2)

	end
	max_motif_match = min(motifA ,motifB)
    best_matched_motifs = -1
    best_i  = -1
	best_j = -1
	best_matching = Dict{Int,Int}()

	m = A.n
	n = B.n

	results = Dict(
		"TAME_timings" => Array{Float64,1}(undef,length(alphas)*length(betas)),
		"Matching Timings"=> Array{Float64,1}(undef,length(alphas)*length(betas)),
		"Scoring Timings"=> Array{Float64,1}(undef,length(alphas)*length(betas)),
	)
	exp_index = 1

    U = Array{Float64,2}(undef,m,iter)
    V = Array{Float64,2}(undef,n,iter)
	best_U = Array{Float64,2}(undef,m,iter)
	best_V = Array{Float64,2}(undef,n,iter)


    for α in alphas
        for beta in betas

			((U,V),runtime) = @timed ΛTAME(A,B,beta,iter,tol,α)
			results["TAME_timings"][exp_index] = runtime

			if typeof(matchingMethod) === ΛTAME_rankOneMatching
				((matched_motifs, i, j, matching,scoring_time),runtime) = @timed search_Krylov_space(A,B,U,V;returnScoringTimings=returnTimings(),kwargs...)
				results["Matching Timings"][exp_index] = runtime - scoring_time
				results["Scoring Timings"][exp_index] = scoring_time
			elseif typeof(matchingMethod) === ΛTAME_GramMatching
				(matched_motifs, gaped_motifs, matching, matching_time, scoring_time)= TAME_score(A,B,U*V';return_timings=returnTimings(),kwargs...)	
				i = -1
				j = -1
				best_U = copy(U)
				best_V = copy(V)
				results["Matching Timings"][exp_index] = matching_time
				results["Scoring Timings"][exp_index] = scoring_time
			else 
				throw(TypeError("typeof MatchingMethod must be either ΛTAME_rankOneMatching or ΛTAME_GramMatching"))
			end

			
			exp_index +=1 

			if matched_motifs > best_matched_motifs
				best_matched_motifs = matched_motifs
				best_i = i
				best_j = j
				best_U = U
				best_V = V
				best_matching = matching
			end
			
			println("α:$(α) -- β:$(beta) finished -- motif_match:$matched_motifs -- max_motif $max_motif_match -- best motif_match: $best_matched_motifs")
        end
    end

	println("best i:$best_i -- best j:$best_j")
	if typeof(matchingMethod) === ΛTAME_rankOneMatching
		return ΛTAME_Return(best_matched_motifs,(motifA ,motifB),
						    best_matching,(best_i, best_j),(best_U, best_V),results)
	else
		return ΛTAME_Return(best_matched_motifs,(motifA ,motifB),
							best_matching,nothing,(best_U, best_V),results)
	end

end


function ΛTAME_param_search_profiled(A_tensors::Array{SymTensorUnweighted{S},1},B_tensors::Array{SymTensorUnweighted{S},1};
							iter::Int = 15,tol::Float64=1e-6,
							alphas::Array{F,1}=[.5,1.0],
							betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
							matchingMethod::MatchingMethod=ΛTAME_rankOneMatching(),kwargs...) where {F <: AbstractFloat, S <: Motif}

	motifCountA = [size(A.indices,2) for A in A_tensors]
	motifCountB = [size(B.indices,2) for B in B_tensors]
	max_motif_match = [min(A_motifCount,B_motifCount) for (A_motifCount,B_motifCount) in zip(motifCountA,motifCountB)]

	m = maximum([A.n for A in A_tensors])
	n = maximum([B.n for B in B_tensors])

	U = Array{Float64,2}(undef,m,iter)
	V = Array{Float64,2}(undef,n,iter)
	best_U = Array{Float64,2}(undef,m,iter)
	best_V = Array{Float64,2}(undef,n,iter)

	profiling = Dict(
		"TAME_timings" => Array{Float64,1}(undef,length(alphas)*length(betas)),
		"Matching Timings"=> Array{Float64,1}(undef,length(alphas)*length(betas)),
		"Scoring Timings"=> Array{Float64,1}(undef,length(alphas)*length(betas)),
	)
	exp_index = 1

	best_matching_score = -1
	best_i  = -1
	best_j = -1
	best_matched_motifs::Array{Int,1} = []
	best_matching = Dict{Int,Int}()

	for α in alphas
		for beta in betas

			(U,V),ΛTAME_runtime = @timed ΛTAME(A_tensors,B_tensors,beta,iter,tol,α)
			profiling["TAME_timings"][exp_index] = ΛTAME_runtime 
			
			if typeof(matchingMethod) === ΛTAME_rankOneMatching
				((matching_score, matched_motifs, i, j, matching,scoring_time), full_runtime) = @timed search_Krylov_space(A_tensors,B_tensors,U,V;returnScoringTimings=returnTimings())
				profiling["Matching Timings"][exp_index] = full_runtime - scoring_time
				profiling["Scoring Timings"][exp_index] = scoring_time 
			elseif typeof(matchingMethod) === ΛTAME_GramMatching
				((matching_score, matched_motifs, matching, matching_time, scoring_time),matching_time) = @timed TAME_score(A_tensors,B_tensors,U*V';return_timings=returnTimings(),kwargs...)
				i = -1
				j = -1
				profiling["Matching Timings"][exp_index] = matching_time
				profiling["Scoring Timings"][exp_index] = scoring_time
			else 
				throw(TypeError("typeof MatchingMethod must be either ΛTAME_rankOneMatching or ΛTAME_GramMatching"))
			end

			exp_index +=1 

			if matching_score > best_matching_score
				best_matching_score = matching_score
				best_matched_motifs = matched_motifs
				best_i = i
				best_j = j
				best_U = U
				best_V = V
				best_matching = matching
			end
			println("α:$(α) -- β:$(beta) finished -- motif_match:$matched_motifs -- max_motifs:$max_motif_match -- matching_score:$matching_score -- best score:$best_matching_score")
			
		end
	end
	
	println("best i:$best_i -- best j:$best_j")
	if typeof(matchingMethod) === ΛTAME_rankOneMatching
		return ΛTAME_MultiMotif_Return(best_matching_score,best_matched_motifs, (motifCountA,motifCountB),
					best_matching,(best_i, best_j),(best_U, best_V),profiling)
	else
		return ΛTAME_MultiMotif_Return(best_matching_score,best_matched_motifs, (motifCountA,motifCountB),
					best_matching,nothing,(best_U, best_V),profiling)
	end

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
    V[:,1] /=norm(V[:,1])

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


function ΛTAME(A::SymTensorUnweighted{S}, B::SymTensorUnweighted{S}, β::Float64,
	max_iter::Int,tol::Float64,α::Float64;update_user=-1) where {S <: Motif}

	U = zeros(A.n,max_iter+1)
	V = zeros(B.n,max_iter+1) #store initial in first column

	U[:,1] = ones(A.n)
	U[:,1] /=norm(U[:,1])

	V[:,1] = ones(B.n)
	V[:,1] /=norm(V[:,1])

	sqrt_β = β^(.5)

	lambda = Inf
	i = 1

	A_buf = zeros(A.n)
	B_buf = zeros(B.n)

	while true

		DistributedTensorConstruction.contraction!(A,U[:,i],A_buf)
		DistributedTensorConstruction.contraction!(B,V[:,i],B_buf)

		U[:,i+1] .= A_buf
		V[:,i+1] .= B_buf


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
			A_buf .= 0.0 
			B_buf .= 0.0 
			i += 1
		end

	end

end

function ΛTAME(A_tensors::Array{SymTensorUnweighted{S},1}, B_tensors::Array{SymTensorUnweighted{S},1}, β::Float64,
	max_iter::Int,tol::Float64,α::Float64;update_user=-1) where {S <: Motif}

	@assert length(A_tensors) == length(B_tensors)
	for i =1:length(A_tensors)
		@assert A_tensors[i].order == B_tensors[i].order
	end

	m = maximum([tensor.n for tensor in A_tensors])
	n = maximum([tensor.n for tensor in B_tensors])

	U = zeros(m,max_iter+1)
	V = zeros(n,max_iter+1) #store initial in first column

	U[:,1] = ones(m)
	U[:,1] /=norm(U[:,1])

	V[:,1] = ones(n)
	V[:,1] /=norm(V[:,1])

	sqrt_β = β^(.5)

	lambda = Inf
	i = 1

	A_buf = zeros(m)
	B_buf = zeros(n)

	while true

		#DistributedTensorConstruction.embedded_contraction!(A_tensors,U[:,i],A_buf)
		#DistributedTensorConstruction.embedded_contraction!(B_tensors,V[:,i],B_buf)
		embedded_contraction!(A_tensors,U[:,i],A_buf)
		embedded_contraction!(B_tensors,V[:,i],B_buf)

		#contraction_divide_out!(A_tensors,U[:,i],A_buf)
		#contraction_divide_out!(B_tensors,V[:,i],B_buf)

		U[:,i+1] .= A_buf
		V[:,i+1] .= B_buf


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
			A_buf .= 0.0 
			B_buf .= 0.0 
			i += 1
		end

	end

end