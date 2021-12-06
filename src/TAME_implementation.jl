
#TODO: convert two function calls into Union{COOTen,ThirdOrderSymTensor}
#TODO: update new DistributedTensorConstructio types in docs
struct TAME_Return{T} <: returnType
	matchScore::Int
	motifCounts::Tuple{Int,Int}
	matching::Union{Dict{Int,Int},Vector{Int}}
	embedding::Matrix{T}
	profile::Union{Nothing,Vector{Tuple{String,Dict{String,Union{Array{Float64,1},Array{Array{Float64,1},1}}}}}}
end

#=------------------------------------------------------------------------------
              Routines for searching over alpha/beta parameters
------------------------------------------------------------------------------=#

#add in SparseSymmetricTensors.jl function definitions
function TAME_param_search(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
                           iter::Int = 15,tol::Float64=1e-6, alphas::Array{F,1}=[.5,1.0],
						   betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						   kwargs...) where {F <: AbstractFloat}
	A_motifs = size(A.indices,1)
	B_motifs = size(B.indices,1)
	max_triangle_match = min(A_motifs,B_motifs)
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


	return TAME_Return(best_TAME_PP_tris,(A_motifs,B_motifs),best_matching,best_TAME_PP_x,nothing)
end

function TAME_param_search_profiled(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor;
                                    iter::Int = 15,tol::Float64=1e-6, alphas::Array{F,1}=[.5,1.0],
						            betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						            profile::Bool=false,profile_aggregation="all",
						            kwargs...) where {F <: AbstractFloat}
	A_motifs = size(A.indices,1)
	B_motifs = size(B.indices,1)
	max_triangle_match = min(A_motifs,B_motifs)
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


	return TAME_Return(best_TAME_PP_tris,(A_motifs,B_motifs),best_matching,best_TAME_PP_x,experiment_profiles)

end

function TAME_param_search(A::SymTensorUnweighted{M},B::SymTensorUnweighted{M};
                           iter::Int = 15,tol::Float64=1e-6, alphas::Array{F,1}=[.5,1.0],
						   betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						   kwargs...) where {F <: AbstractFloat,M <:Motif}
	A_motifs = size(A.indices,2)
	B_motifs = size(B.indices,2)
	max_motif_match = min(A_motifs,B_motifs)
	best_TAME_PP_motif::Int = -1
	best_matching = Dict{Int,Int}()

	m = A.n
	n = B.n

	best_TAME_PP_x = Array{Float64,2}(undef,m,n)


    for α in alphas
        for β in betas

			x, motif_count, matching = TAME(A,B,β,iter,tol,α;W = ones(A.n,B.n),kwargs...)

			if motif_count > best_TAME_PP_motif
				best_matching = matching
                best_TAME_PP_motif = motif_count
				best_TAME_PP_x = copy(x)
            end
			println("α:$(α) -- β:$β finished -- tri_match:$(motif_count) -- max_tris $(max_motif_match) -- best tri_match:$(best_TAME_PP_motif)")
        end

    end


	return TAME_Return(best_TAME_PP_motif,(A_motifs,B_motifs),best_matching,best_TAME_PP_x,nothing)
end

function TAME_param_search_profiled(A::SymTensorUnweighted{M},B::SymTensorUnweighted{M};
                                    iter::Int = 15,tol::Float64=1e-6, alphas::Array{F,1}=[.5,1.0],
						            betas::Array{F,1} =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001],
						            profile::Bool=false,profile_aggregation="all",
						            kwargs...) where {F <: AbstractFloat, M <: Motif}
	A_motifs = size(A.indices,2)
	B_motifs = size(B.indices,2)
	max_motif_match = min(A_motifs,B_motifs)
    total_motif = size(A.indices,1) + size(B.indices,1)
	best_TAME_PP_motif = -1
	best_matching = Dict{Int,Int}()

	m = A.n
	n = B.n

	best_TAME_PP_x = Array{Float64,2}(undef,m,n)
	experiment_profiles = Array{Tuple{String,Dict{String,Union{Array{Float64,1},Array{Array{Float64,1},1}}}},1}(undef,0)

    for α in alphas
        for β in betas

			x, motif_count, matching, experiment_profile = TAME_profiled(A,B,β,iter,tol,α;W = ones(m,n),kwargs...)
			push!(experiment_profiles,("α:$(α)_β:$(β)",experiment_profile))

			if motif_count > best_TAME_PP_motif
				best_matching = matching
                best_TAME_PP_motif = motif_count
				best_TAME_PP_x = copy(x)
            end
			println("α:$(α) -- β:$β finished -- tri_match:$(best_TAME_PP_motif) -- max_motifs $(max_motif_match)")
        end

    end


	return TAME_Return(best_TAME_PP_motif,(A_motifs,B_motifs),best_matching,best_TAME_PP_x,experiment_profiles)

end

#=------------------------------------------------------------------------------
             		    Spectral Relaxation Routines
------------------------------------------------------------------------------=#


function setup_tame_data(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor)
	return _index_triangles_nodesym(A.n,A.indices), _index_triangles_nodesym(B.n,B.indices)
end

function setup_tame_data(A::SymTensorUnweighted{M},B::SymTensorUnweighted{M}) where {M <: Motif}
	return _index_motifs_nodesym(A.n,A.indices), _index_motifs_nodesym(B.n,B.indices)
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

function _index_motifs_nodesym(n,Motifs::Array{Int,2})

	free_modes = collect(size(Motifs,1):-1:1)
	contracted_modes = collect(combinations(1:size(Motifs,1),size(Motifs,1)-1))

	Mi = [Vector{Vector{Int}}(undef, 0) for i in 1:n ]
	for edge in eachcol(Motifs)
		for (free_mode,contracted_mode) in zip(free_modes,contracted_modes)
			push!(Mi[edge[free_mode]], edge[contracted_mode])
		end

	end
	sort!.(Mi)
	return Mi
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
	* 'best_motif_count' - (Int)
	  The maximum number of triangles matched. 
	* 'best_matching' - (Dict{Int,Int})
	  The matching computed between the two graphs, maps the vertices of A to the
	  vertices of B. 
------------------------------------------------------------------------------"""
function TAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,β::F, max_iter::Int,
			  tol::F,α::F;update_user::Int=-1,W::Array{F,2}=ones(A.n,B.n),
			  no_matching=false,kwargs...) where {F <:AbstractFloat}

    dimension = minimum((A.n,B.n))

	A_Ti, B_Ti = setup_tame_data(A,B)

	best_x = Array{Float64,2}(undef,A.n,B.n)
    best_motif_count = -1
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
			triangles, missed_motifs, matching =  TAME_score(A,B,X;kwargs...)

			if update_user != -1 && i % update_user == 0
				println("finished iterate $(i):tris:$triangles -- gaped_t:$missed_motifs")
			end

			if triangles > best_motif_count
				best_matching = matching
				best_x = copy(x_k_1)
				best_motif_count = triangles
				best_iterate = i
			end

		end

		if update_user != -1 && i % update_user == 0
			println("λ: $(new_lambda)")
		end


        if abs(new_lambda - lambda) < tol || i >= max_iter
  			return reshape(best_x,A.n,B.n), best_motif_count, best_matching
        else
            x_k = copy(x_k_1)
            lambda = new_lambda
            i += 1
        end

    end


end

function TAME(A::SymTensorUnweighted{M}, B::SymTensorUnweighted{M}, β::F, max_iter::Int,
	                   tol::F,α::F;update_user::Int=-1,W::Array{F,2} = ones(A.n,B.n),
					   no_matching::Bool=false,kwargs...) where {F <:AbstractFloat, M <: Motif}

 
	A_Mi, B_Mi = setup_tame_data(A,B)

	best_X = Array{Float64,2}(undef,A.n,B.n)
    best_motif_count::Int = -1
	best_matching = Dict{Int,Int}()

    X_k = copy(W)
	X_k ./=norm(X_k)

	if α != 1.0 
		X_0_with_alpha = copy(X_k)*(1-α)
	end


    i = 1
    lambda = Inf

    while true

		X_k_1 = impTTVnodesym(A.n, B.n, size(A.indices,1), X_k, A_Mi, B_Mi)

        new_lambda = dot(X_k_1,X_k)

        if β != 0.0
            X_k_1 .+= β * X_k
        end

        if α != 1.0
            X_k_1 .*=  α
			X_k_1 .+= X_0_with_alpha
        end

        X_k_1 ./= norm(X_k_1)

		if !no_matching

			matched_motifs, missed_motifs, matching = TAME_score(A,B,X_k_1;kwargs...)
			
			if update_user != -1 && i % update_user == 0
				println("finished iterate $(i): matched motifs:$(matched_motifs) -- missed motifs:$(missed_motifs)")
			end

			if matched_motifs > best_motif_count
				best_matching = matching
				best_X = copy(X_k_1)
				best_motif_count = matched_motifs
			end
		end

		if update_user != -1 && i % update_user == 0
			println("λ_$i: $(new_lambda)")
		end

        if abs(new_lambda - lambda) < tol || i >= max_iter
   			return best_X, best_motif_count, best_matching
        else
            X_k = copy(X_k_1)
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
	* 'best_motif_count' - (Int)
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
	  +'scoring_timings' - Time taken to score each matching. 

------------------------------------------------------------------------------"""
function TAME_profiled(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor, β::F, max_iter::Int,
	                   tol::F,α::F;update_user::Int=-1,W::Array{F,2} = ones(A.n,B.n),
					   no_matching::Bool=false,kwargs...) where {F <:AbstractFloat}

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
    best_motif_count::Int = -1
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

			triangles, missed_motifs, matching, matching_time, scoring_time = TAME_score(A,B,reshape(x_k_1,A.n,B.n);return_timings=returnTimings(),kwargs...)
			
			push!(experiment_profile["matching_timings"],matching_time)
			push!(experiment_profile["scoring_timings"],scoring_time)
			push!(experiment_profile["matched triangles"],float(triangles))
			push!(experiment_profile["gaped triangles"],float(missed_motifs))


			if update_user != -1 && i % update_user == 0
				println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(missed_motifs)")
			end

			if triangles > best_motif_count
				best_matching = matching
				best_x = copy(x_k_1)
				best_motif_count = triangles
				best_iterate = i
			end
		end

		if update_user != -1 && i % update_user == 0
			println("λ_$i: $(new_lambda)")
		end

        if abs(new_lambda - lambda) < tol || i >= max_iter
   			return reshape(best_x,A.n,B.n), best_motif_count, best_matching, experiment_profile
        else
            x_k = copy(x_k_1)
            lambda = new_lambda
            i += 1
        end

    end


end



# currently using types on A and B to alert user against passing Vector{SymTensorUnweighted{M}} in. 
function TAME_profiled(A::SymTensorUnweighted{M}, B::SymTensorUnweighted{M}, β::F, max_iter::Int,
	                   tol::F,α::F;update_user::Int=-1,W::Array{F,2} = ones(A.n,B.n),
					   no_matching::Bool=false,kwargs...) where {F <:AbstractFloat, M <: Motif}

    dimension = minimum((A.n,B.n))

    experiment_profile = Dict{String,Union{Array{F,1},Array{Array{F,1}}}}(
		"contraction_timings"=>Array{F,1}(undef,0),
		"matching_timings"=>Array{F,1}(undef,0),
		"scoring_timings"=>Array{F,1}(undef,0),
		"svd_timings"=>Array{F,1}(undef,0),
		"matched motifs"=>Array{F,1}(undef,0),
		"missed motifs"=>Array{F,1}(undef,0),
		"sing_vals"=>Array{Array{F,1},1}(undef,0),
		"ranks"=>Array{F,1}(undef,0)
	)


	A_Mi, B_Mi = setup_tame_data(A,B)

	best_X = Array{Float64,2}(undef,A.n,B.n)
    best_motif_count::Int = -1
	best_index = -1
	best_matching = Dict{Int,Int}()

    X_k = copy(W)
	X_k ./=norm(X_k)

	if α != 1.0 
		X_0_with_alpha = copy(X_k)*(1-α)
	end


    i = 1
    lambda = Inf

    while true

		X_k_1,t = @timed impTTVnodesym(A.n, B.n, size(A.indices,1), X_k, A_Mi, B_Mi)
		push!(experiment_profile["contraction_timings"],t)

        new_lambda = dot(X_k_1,X_k)

        if β != 0.0
            X_k_1 .+= β * X_k
        end

        if α != 1.0
            X_k_1 .*=  α
			X_k_1 += X_0_with_alpha
        end

        X_k_1 ./= norm(X_k_1)

		S,t = @timed svdvals(X_k_1)
		push!(experiment_profile["svd_timings"],t)

		rank = 0.0
		for i in 1:length(S)
			if S[i] > S[1]*eps(Float64)*dimension
				rank = rank + 1
			end
		end

		push!(experiment_profile["sing_vals"],S)
		push!(experiment_profile["ranks"],rank)

		if !no_matching

			matched_motifs, missed_motifs, matching, matching_time, scoring_time = TAME_score(A,B,X_k_1;return_timings=returnTimings(),kwargs...)
			
			push!(experiment_profile["matching_timings"],matching_time)
			push!(experiment_profile["scoring_timings"],scoring_time)
			push!(experiment_profile["matched motifs"],float(matched_motifs))
			push!(experiment_profile["missed motifs"],float(missed_motifs))


			if update_user != -1 && i % update_user == 0
				println("finished iterate $(i):tris:$(matched_motifs) -- gaped_t:$(missed_motifs)")
			end

			if matched_motifs > best_motif_count
				best_matching = matching
				best_X = copy(X_k_1)
				best_motif_count = matched_motifs
			end
		end

		if update_user != -1 && i % update_user == 0
			println("λ_$i: $(new_lambda)")
		end

        if abs(new_lambda - lambda) < tol || i >= max_iter
   			return best_X, best_motif_count, best_matching, experiment_profile
        else
            X_k = copy(X_k_1)
            lambda = new_lambda
            i += 1
        end

    end


end