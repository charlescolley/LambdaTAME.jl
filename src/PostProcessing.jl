abstract type PostProcessingMethod end
struct noPostProcessing <: PostProcessingMethod end

@with_kw struct KlauAlgo <: PostProcessingMethod 
    k::Int = -1;
    a::Int = 1;
	b::Int = 1;
	stepm::Int = 25;
	rtype::Int = 1;
	maxiter::Int = 500;
	verbose::Bool = false;
	gamma::Float64 =.4;
end

abstract type SKUpdateRule end
struct constant_R <: SKUpdateRule end
struct linear_R <: SKUpdateRule end
struct fgap_R <: SKUpdateRule end
struct overlap_R <: SKUpdateRule end
struct edges_R <: SKUpdateRule end

@with_kw struct SuccessiveKlauAlgo <: PostProcessingMethod 
    k::Int = -1;
	iterDelta::Int = 100;
	verbose::Bool = true;
    update_rule::SKUpdateRule = overlap_R();
    maximum_Klau_iters::Int=1000;
    successive_iter::Int=10;
    start_maxIter::Int = 100;
end



"""-----------------------------------------------------------------------------
   Greedily creates a b matching from a rank 1 bipartite matching problem. Only
  adds pairs to the matchings if they have the same sign, and neither edge value
  is zero.

  Output:
  -------
  * Matchings - (2D Int Array):
    The ith row corresponds to the b-matchings of the ith vertex represented by
    the vector u to the vertices in v. unmatched vertices are set to -1.

-----------------------------------------------------------------------------"""
#TODO: This version is outdated, should be removed and replaced
#      with 'rank_one_bmatching' below
function b_matching(u::Array{Float64,1}, v::Array{Float64,1}, b::Int)

    @assert length(u) <= length(v) #make first vector shorter

    n = length(u)

    #get indices sorted by increasing order
    u_indices = sort(1:length(u),by=i->-u[i])
    v_indices = sort(1:length(v),by=i->-v[i])

    matching_degrees = zeros(length(v))

    Matchings = Array{Tuple{Int,Int,Float64},1}(undef,b*n)  #-1 indicates unset matching
    match_idx = 1
    starting_index = 1
    ending_index = b

    for i in 1:n

        if u[i] == 0.0
            continue
        else
            #check matching degree of starting index
            while matching_degrees[v_indices[starting_index]] >= b
                starting_index += 1
            end
            ending_index = minimum((starting_index + b,length(v)))
            #check that sign of ending vertex doesn't change
            while v[v_indices[ending_index]]*u[u_indices[i]] <= 0
                ending_index -= 1
            end

            @assert ending_index != starting_index  #make sure something's added

            for j in 1:(ending_index - starting_index)
                v_j = v_indices[starting_index + (j-1)]
                Matchings[match_idx] = (u_indices[i],v_j,v[v_j]*u[u_indices[i]])
                match_idx += 1
                matching_degrees[v_j] += 1
            end
        end
    end

    return Matchings[1:(match_idx-1)]



@with_kw struct TabuSearch <: PostProcessingMethod 
    k::Int = -1;
	iterations::Int=10;
	verbose::Bool = true;
end


function rank_one_bmatching(u::Array{T,1},v::Array{T,1},b::Int) where T
	#current implementation will end up rounding b to be odd


    #get maximum matching by rearrangement theorem
    u_perm = sortperm(u,rev=true)
    v_perm = sortperm(v,rev=true)

    #matching_weight = zero(T )
    Match_candidates = Set()
    #for (i,j) in zip(u_perm,v_perm)
	for i = 1:length(u_perm)

		u_i = u_perm[i]
		#look at a size b window centered around 
		for offset = ceil(-b/2):1:floor(b/2)
			
			j = i + Int(offset)
			#println(j == i)
			#println("i:$i  j:$j")
			#print(offset)
			if j >= 1 && j <= length(v_perm)
				v_j = v_perm[j]
				#only match positive to positive and negative to negative
				if (u[u_i] > 0 && v[v_j] > 0 ) || (u[u_i] < 0 && v[v_j] < 0)
					push!(Match_candidates,(u_i,v_j))
				end
			end
		end
    end
    return Match_candidates
end


"""-----------------------------------------------------------------------------
    Computes the cost of swapping the matching of (i,j) & (k,l) to (i,l) & (k,j)
  in terms of the number of triangles. Inputs are the triangles associated with
  each of the vertices

  Inputs:
  -------
  * T_{i,j,k,l} -(Set{Array{Int,1}}):
    All triangles incident with the vertices i,j,k,l .
  * original_matching - (Dict{Int,Int})
    The original matching from the vertices in graph A to the vertices in graph
    B.

-----------------------------------------------------------------------------"""
function compute_triangles_aligned(T_i::Set{Array{Int,1}},T_j::Set{Array{Int,1}},
  #                                  T_k::Set{Array{Int,1}},T_l::Set{Array{Int,1}},
                                    M::Dict{Int,Int})
    #compute the original number of aligned triangles
    matched_tris = 0
    for (v_i,v_j,v_k) in T_i
        if sort([get(M,v_i,-1),get(M,v_j,-1),get(M,v_k,-1)]) in T_j
            matched_tris += 1
        end
    end

    #=
    for (v_i,v_j,v_k) in T_k
        if sort([get(M,v_i,-1),get(M,v_j,-1),get(M,v_k,-1)]) in T_l
            matched_tris += 1
        end
    end


    #compute the new matched_triangles end
    new_matched_tris = 0
    for (v_i,v_j,v_k) in T_i
        if sort([get(M,v_i,-1),get(M,v_j,-1),get(M,v_k,-1)]) in T_l
            new_matched_tris += 1
        end
    end
    for (v_i,v_j,v_k) in T_k
        if sort([get(M,v_i,-1),get(M,v_j,-1),get(M,v_k,-1)]) in T_j
            new_matched_tris += 1
        end
    end
    =#
    return matched_tris#, new_matched_tris
end

"""-----------------------------------------------------------------------------
     Finds all the triangles associated with the vertices and preprocesses them
   into a list of Sets. Used to preprocess the searching for triangles
   associated with a vertex.

   Inputs:
   -------
   * triangles - (Array{Int,2}):
     The triangles contained within the graph. indices in each row are expected
     to be sorted in ascending order.
   * n - (Int):
     The number of vertices in the graph.

   Outputs:
   --------
   * incidence - (Array{Set{Array{Int,1}}},1):
     An array where the ith entry contains a set linking the vertex to the
     triangles it's contained within.
-----------------------------------------------------------------------------"""
function produce_triangle_incidence(triangles::Array{Int,2},n::Int)
    incidence = Array{Set{Array{Int,1}},1}(undef,n)
    for i in 1:n
        incidence[i] = Set{Array{Int,1}}()
    end

    for t in eachrow(triangles)
        for v in t
            push!(incidence[v],t)
        end
    end

    return incidence
end

"""-----------------------------------------------------------------------------
                    Klau's algorithm PostProcessing
-----------------------------------------------------------------------------"""

function netalignmr(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
                    U::Matrix{T},V::Matrix{T},params::KlauAlgo;kwargs...) where T
    (_,_,matching,_)=  bipartite_matching_primal_dual(U*V';kwargs...)
    return netalignmr(A,B,knearest_sparsification(U,V,matching_array_to_dict(matching),2*min(size(U,2),size(V,2))),params)
end

function netalignmr(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
                    U::Matrix{T},V::Matrix{T},k::Int,params::KlauAlgo;kwargs...) where T
    (_,_,matching,_)=  bipartite_matching_primal_dual(U*V';kwargs...)
    return netalignmr(A,B,knearest_sparsification(U,V,matching_array_to_dict(matching),k),params)
end

function netalignmr(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
                    U::Matrix{T},V::Matrix{T},
                    matching::Union{Vector{NTuple{2,Int}},Dict{Int,Int}},k::Int,params::KlauAlgo) where T
    return netalignmr(A,B,knearest_sparsification(U,V,matching,k),params)
end

function netalignmr(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
                    U::Matrix{T},V::Matrix{T},
                    matching::Union{Vector{NTuple{2,Int}},Dict{Int,Int}},params::KlauAlgo) where T
    return netalignmr(A,B,knearest_sparsification(U,V,matching,2*min(size(U,2),size(V,2))),params)
end


function netalignmr(A::SparseMatrixCSC{T1,Int},B::SparseMatrixCSC{T1,Int},L::SparseMatrixCSC{T2,Int},params::KlauAlgo) where {T1,T2}

	@assert size(L,1) == size(A,1) && size(L,2) == size(B,1)

	#L = sparse(X)
	S,w,li,lj = netalign_setup(A,B,L)

	# align networks
	(xbest,st,status,hist), t_netalignmr = @timed NetworkAlign.netalignmr(S,w,params.a,params.b,li,lj,params.gamma,params.stepm, 
                                                                          params.rtype,params.maxiter, params.verbose)
    final_link_L = sparse(li,lj,xbest,size(A,1),size(B,1))
    dropzeros!(final_link_L)

    matching = MatrixNetworks.edge_list(NetworkAlign.bipartite_matching(final_link_L))

    matching = Dict(zip(matching...))

	return matching, t_netalignmr, nnz(L)/(size(A,1)*size(B,1)), status


end
#=
function successive_netalignmr(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
                               U::Matrix{T},V::Matrix{T},
                               matching::Union{Vector{NTuple{2,Int}},Dict{Int,Int}}
                               ,params::successiveKlauAlgo) where T
    
    if typeof(matching) === Vector{NTuple{2,Int}}
        matching = Dict(matching)
    end

	(prev_matching,_,sparsity),first_run = @timed netalignmr(A,B,U,V,matching_array_to_dict(matching),size(U,2),params)
                                    #using size(U,2) as a stand in for rank(U)

    timings = [first_run]
    sparsities = [sparsity]
	cur_maxiter = 50
    maximum_iters=1000
	prev_match_change = 0
	for i = 1:10
		(klau_matching,_,sparsity),t = @timed netalignmr(A,B,U,V,prev_matching,2*15,KlauAlgo(maxiter=cur_maxiter))
        push!(timings,t)
        push!(sparsities,sparsity)
        match_change = length(intersect(Set(klau_matching),Set(prev_matching)))

		println("matchings changed by $(match_change - prev_match_change)")
		if cur_maxiter < maximum_iters
			if match_change <= 15
				cur_maxiter *= 2
				println("new matching similar to previous one, upping Klau iterations to $cur_maxiter")
			end
		end


		if klau_matching == match_change
			println("matching didn't change")
		end
		
		prev_match_change = match_change
		prev_matching = copy(klau_matching)
	end
    return klau_matching, timings, sparsities
end
=#
function successive_netalignmr_profiled(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
                                        U::Matrix{T},V::Matrix{T},
                                        m0::Union{Vector{NTuple{2,Int}},Dict{Int,Int}},
                                        params::SuccessiveKlauAlgo) where T

    if typeof(m0) === Vector{NTuple{2,Int}}
        m0 = Dict(m0)
    end

    overlap_check = Int(ceil(.05*min(size(A,1),size(B,1))))
	edge_match = matching -> edges_matched(A,B,matching)[1]/2

	#  --  Build first link matrix  --  #
    T_U,t_u = @timed BallTree(U')
	T_V,t_v = @timed BallTree(V')
    kkn_rt = t_u + t_v
	
	L0,t  = @timed knearest_sparsification(T_U,T_V,U,V,m0,size(U,2)) 
    kkn_rt += t

	#run one call of Klau like normal
	klau_params = KlauAlgo()
	(old_matching,_,_,status),rt = @timed netalignmr(A,B,L0,klau_params)
	
	matching_stats = Dict([
		("edges_matched",[edge_match(old_matching)]),
		("runtime",[rt]),
		("kkn L runtime",[kkn_rt]),
		("iters_used",[klau_params.maxiter]),
		("match_overlap",[-1]),
		("status",[status])
	])

	f_gap = matching_stats["status"][end][2] - matching_stats["status"][end][1]
    if params.verbose
	    println("iter 0:rt:$(matching_stats["runtime"][end]) -- match_overlap:$(matching_stats["match_overlap"][end]) -- edges_mapped:$(matching_stats["edges_matched"][end]) -- status:$(matching_stats["status"][end]) -- f_gap:$(f_gap)")
    end
    cur_maxiter = params.start_maxIter
	
	prev_check_val = 0
	nextTimeBreak=false
	
	best_edgeMatch = -1
	best_matching = Dict{Int,Int}()
	for i = 1:params.successive_iter

		L,t = @timed knearest_sparsification(T_U,T_V,U,V,old_matching,size(U,2)) 
		push!(matching_stats["kkn L runtime"],t)
		(klau_matching,_,_,status),rt = @timed netalignmr(A,B,L,KlauAlgo(maxiter=cur_maxiter))
		
		push!(matching_stats["runtime"],rt)
		push!(matching_stats["match_overlap"],length(intersect(Set(klau_matching),Set(old_matching))))
		push!(matching_stats["edges_matched"],edge_match(klau_matching))
		push!(matching_stats["iters_used"],cur_maxiter)
		push!(matching_stats["status"],status)

		fgap = matching_stats["status"][end][2] - matching_stats["status"][end][1]
        if params.verbose
    		println("iter $i: rt:$(matching_stats["runtime"][end]) -- match_overlap:$(matching_stats["match_overlap"][end]) -- edges_mapped:$(matching_stats["edges_matched"][end]) -- status:$(matching_stats["status"][end]) -- f_gap:$(fgap)")
        end

		if cur_maxiter < params.maximum_Klau_iters

			if params.update_rule === fgap_R()
				should_miter_increase = fgap > prev_check_val
			elseif params.update_rule === edges_R()
				should_miter_increase = matching_stats["edges_matched"][end] < prev_check_val
			elseif params.update_rule === overlap_R()
				should_miter_increase = ((matching_stats["match_overlap"][end] - prev_check_val) <= overlap_check)
			elseif params.update_rule === linear_R()
				should_miter_increase = true
			elseif params.update_rule == constant_R()
				should_miter_increase = false
			end

			if should_miter_increase
				cur_maxiter += params.iterDelta
                if params.verbose
				    println("upping Klau iterations to $cur_maxiter")
                end
			end
		end

		if matching_stats["edges_matched"][end] >= best_edgeMatch
			best_edgeMatch = matching_stats["edges_matched"][end]
			best_matching = copy(klau_matching)
		end

		if length(old_matching) == matching_stats["match_overlap"][end] 
            if params.verbose
			    print("matching didn't change")
            end
            if nextTimeBreak
                if params.verbose
				    println(" twice in a row, breaking...")
                end
				break
			else
                if params.verbose
    				print("\n")
                end
				nextTimeBreak=true
			end
		else
			nextTimeBreak=false
		end
		
		if params.update_rule === fgap_R()
			prev_check_val = fgap
		elseif params.update_rule === edges_R()
			prev_check_val = matching_stats["edges_matched"][end]
		elseif params.update_rule === overlap_R()
			prev_check_val = matching_stats["match_overlap"][end]
		end

		old_matching = copy(klau_matching)
	end
	return best_matching, matching_stats

end

function successive_netalignmr(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
                               U::Matrix{T},V::Matrix{T},
                               m0::Union{Vector{NTuple{2,Int}},Dict{Int,Int}},
                               params::SuccessiveKlauAlgo) where T

    if typeof(m0) === Vector{NTuple{2,Int}}
        m0 = Dict(m0)
    end

    overlap_check = Int(ceil(.05*min(size(A,1),size(B,1))))

	edge_match = matching -> edges_matched(A,B,matching)[1]/2

	#  --  Build first link matrix  --  #
    
	T_U = BallTree(U')
	T_V = BallTree(V')
	L0 = knearest_sparsification(T_U,T_V,U,V,m0,size(U,2)) 

	#run one call of Klau like normal
	klau_params = KlauAlgo()
	old_matching,_,_,status = netalignmr(A,B,L0,klau_params)
	
	f_gap = status[2] - status[1]
    if params.verbose
	    println("iter 0: match_overlap:n/a -- edges_mapped:$(edge_match(old_matching)) -- status:$status -- f_gap:$f_gap")
    end
    cur_maxiter = params.start_maxIter
	
	prev_check_val = 0
	nextTimeBreak=false
	
	best_edgeMatch = -1
	best_matching = Dict{Int,Int}()
	for i = 1:params.successive_iter

		L = knearest_sparsification(T_U,T_V,U,V,old_matching,size(U,2)) 

		klau_matching,_,_,status = netalignmr(A,B,L,KlauAlgo(maxiter=cur_maxiter))
        overlap = length(intersect(Set(klau_matching),Set(old_matching)))
        fgap = status[2] - status[1]
        edges_matched = edge_match(klau_matching)
		
        if params.verbose
            println("iter $i: match_overlap:$overlap -- edges_mapped:$edges_matched -- status:$status -- f_gap:$f_gap")
        end

		if cur_maxiter < params.maximum_Klau_iters

			if params.update_rule === fgap_R()
                fgap = status[2] - status[1]
				should_miter_increase = fgap > prev_check_val
			elseif params.update_rule === edges_R()
				should_miter_increase = edges_matched  < prev_check_val
			elseif params.update_rule === overlap_R()
                overlap = length(intersect(Set(klau_matching),Set(old_matching)))
				should_miter_increase = ((overlap - prev_check_val) <= overlap_check)
			elseif params.update_rule === linear_R()
				should_miter_increase = true
			elseif params.update_rule == constant_R()
				should_miter_increase = false
			end

			if should_miter_increase
				cur_maxiter += params.iterDelta
                if params.verbose
				    println("upping Klau iterations to $cur_maxiter")
                end
			end
		end

		if edges_matched >= best_edgeMatch
			best_edgeMatch = edges_matched
			best_matching = copy(klau_matching)
		end

		if length(old_matching) == overlap 
            if params.verbose
			    print("matching didn't change")
            end
            if nextTimeBreak
                if params.verbose
				    println(" twice in a row, breaking...")
                end
				break
			else
                if params.verbose
    				print("\n")
                end
				nextTimeBreak=true
			end
		else
			nextTimeBreak=false
		end

		#TODO: check to see if this fails
		if params.update_rule === fgap_R()
			prev_check_val = fgap
		elseif params.update_rule === edges_R()
			prev_check_val = edges_matched
		elseif params.update_rule === overlap_R()
			prev_check_val = overlap
		end

		old_matching = copy(klau_matching)
	end
	return best_matching
<<<<<<< HEAD
=======
=======
>>>>>>> 726ef929fb8db472e9b7d74714503910777264ba
>>>>>>> 46378117250e902d5572d97f4fcdb5ac35111ccc

end

function knearest_sparsification(U::Matrix{T},V::Matrix{T},matching::Union{Vector{NTuple{2,Int}},Dict{Int,Int}},k) where T

    @assert maximum([i for (i,j) in matching]) <= size(U,1)
    @assert maximum([j for (i,j) in matching]) <= size(V,1)

	m = size(U,1)
	n = size(V,1)

	candidates = Set()
	T_U = BallTree(U')
	T_V = BallTree(V')

	for (i,j) in matching 
		U_idxs = knn(T_U, U[i,:], minimum((k,m)))[1]
		V_idxs = knn(T_V, V[j,:], minimum((k,n)))[1]

		for ip in U_idxs 
			push!(candidates,(ip,j))
		end

		for jp in V_idxs 
			push!(candidates,(i,jp))
		end
	end

	
	sparse_L = spzeros(m,n)

	for (i,j) in candidates	
		sparse_L[i,j] = U[i,:]'*V[j,:]
	end

	return sparse_L
end

function knearest_sparsification(T_U::BT,T_V::BT,U::Matrix{T},V::Matrix{T},matching::Union{Vector{NTuple{2,Int}},Dict{Int,Int}},k) where {BT <: BallTree,T}

    @assert maximum([i for (i,j) in matching]) <= size(U,1)
    @assert maximum([j for (i,j) in matching]) <= size(V,1)

	m = size(U,1)
	n = size(V,1)

	candidates = Set()

	for (i,j) in matching 
		U_idxs = knn(T_U, U[i,:], minimum((k,m)))[1]
		V_idxs = knn(T_V, V[j,:], minimum((k,n)))[1]

		for ip in U_idxs 
			push!(candidates,(ip,j))
		end

		for jp in V_idxs 
			push!(candidates,(i,jp))
		end
	end

    li = []
    lj = []
    lv = []

	for (i,j) in candidates	
        push!(li,i)
        push!(lj,j)
		push!(lv,U[i,:]'*V[j,:])
	end
	return sparse(li,lj,lv,m,n)
end

function findEmbeddingLookalikes(X::Matrix{S},idx,k) where S
	n = size(X,1)
	#all_nodes = vcat(U,V)
	T = BallTree(X')
	idxs = knn(T, X[idx,:]', minimum((k,n)))[1]
	# form the edges for sparse
	ei = Int[]
	ej = Int[]
	for i=1:length(idx)
	  for j=idxs[i]
		if i != j
		  push!(ei,i)
		  push!(ej,j)
		end
	  end
	end
	return ei,ej
end

#
#  Tabu Search Methods
#


#=
@with_kw struct TabuSearch <: PostProcessingMethod 
    k::Int = -1;
	iterations::Int=10;
	verbose::Bool = true;
end
=#

#function tabu_search_profiled(A, B, A_ten, B_ten, U, V ,matching,max_iters,k,method="new")
function tabu_search_profiled(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
							  A_ten, B_ten, U::Matrix{T},V::Matrix{T},
							  m0::Dict{Int,Int},  params::TabuSearch) where T
	#k = 15
	
	profile_results = Dict{String,Union{Vector{Tuple{Int64,Int64}},Vector{Float64}}}([
		("runtimes",Vector{Float64}(undef,0)),
		("motif match",Vector{Tuple{Int64,Int64}}(undef,0)),
		("edge match",Vector{Tuple{Int64,Int64}}(undef,0)),
		("seq runtimes",Vector{Float64}(undef,0)),
		("top runtimes",Vector{Float64}(undef,0))
	])


	if params.k == -1
		k = 2*size(U,2)
	else
		k = params.k
	end

	if params.verbose
		edges_matched = LambdaTAME.edges_matched(A,B,m0)
		motifs_matched = LambdaTAME.TAME_score(A_ten,B_ten,m0)
		println("iter 0: edges_matched -- $edges_matched  motifs matched -- $motifs_matched")
	end
	#cur_matching = LT_res.matching
	cur_matching = copy(m0)
	for i =1:params.iterations

		(new_matching, seq_scoring_runtimes, top_scoring_runtimes),rt = @timed tabu_search_profiled(A,B,A_ten,B_ten,U,V,cur_matching,k)
		
		push!(profile_results["runtimes"],rt)
		push!(profile_results["seq runtimes"],seq_scoring_runtimes)
		push!(profile_results["top runtimes"],top_scoring_runtimes)

		push!(profile_results["motif match"],LambdaTAME.TAME_score(A_ten,B_ten,new_matching)) 
									#TAME_score wants most motifs in left arg
		push!(profile_results["edge match"],LambdaTAME.edges_matched(A,B,new_matching))
		cur_matching = new_matching

		if params.verbose
			println("iter $i: edges_matched -- $(profile_results["edge match"][end])  motifs matched -- $(profile_results["motif match"][end])  rt -- $rt seq_rt:$seq_scoring_runtimes, top_rt:$top_scoring_runtimes")
		end

	end


	return cur_matching, profile_results
end

function tabu_search(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
							  A_ten, B_ten, U::Matrix{T},V::Matrix{T},
							  m0::Dict{Int,Int},  params::TabuSearch) where T

	if params.k == -1
		k = 2*size(U,2)
	else
		k = params.k
	end

	if params.verbose
		edges_matched = LambdaTAME.edges_matched(A,B,m0)
		motifs_matched = LambdaTAME.TAME_score(A_ten,B_ten,m0)
		println("iter 0: edges_matched -- $edges_matched  motifs matched -- $motifs_matched")
	end
	#cur_matching = LT_res.matching
	cur_matching = copy(m0)
	for i =1:params.iterations

		new_matching = tabu_search(A,B,A_ten,B_ten,U,V,cur_matching,k)
		
		cur_matching = new_matching

		if params.verbose
			edges_matched = LambdaTAME.edges_matched(A,B,new_matching)
			motifs_matched = LambdaTAME.TAME_score(A_ten,B_ten,new_matching)
			println("iter $i: edges_matched -- $edges_matched  motifs matched -- $motifs_matched")
		end

	end


	return cur_matching
end

function create_edge_idx_map(A_ten::SymTensorUnweighted{Clique})

	node_to_edge_indices = Vector{Vector{Int}}(undef,A_ten.n)
	for i in 1:A_ten.n
		node_to_edge_indices[i] = Vector{Int}(undef,0)
	end

	for edge_idx in 1:size(A_ten.indices,2)
		for i in 1:size(A_ten.indices,1)
			ip = A_ten.indices[i,edge_idx]
			push!(node_to_edge_indices[ip],edge_idx)
		end
	end
	return node_to_edge_indices
end

function create_edge_idx_map(A_ten::ThirdOrderSymTensor)

	node_to_edge_indices = Vector{Vector{Int}}(undef,A_ten.n)
	for i in 1:A_ten.n
		node_to_edge_indices[i] = Vector{Int}(undef,0)
	end

	for edge_idx in 1:size(A_ten.indices,1)
		for i in 1:size(A_ten.indices,2)
			ip = A_ten.indices[edge_idx,i]
			push!(node_to_edge_indices[ip],edge_idx)
		end
	end
	return node_to_edge_indices
end




function tabu_search_profiled(A,B,A_ten,B_ten,U,V,matching,k)#,match_method="top_sim")


	m = size(A,1)
	n = size(B,1)

	T_U = BallTree(U')
	T_V = BallTree(V')

	B_motifs = Set{Vector{Int}}()

	if typeof(B_ten) === ThirdOrderSymTensor
		for idx = 1:size(B_ten.indices,1)
			push!(B_motifs,B_ten.indices[idx,:])
		end
	else
		for idx = 1:size(B_ten.indices,2)
			push!(B_motifs,B_ten.indices[:,idx])
		end
	end

	A_node_edge_indices = create_edge_idx_map(A_ten)

	seq_scoring_runtimes = 0.0
	top_scoring_runtimes = 0.0

	matched_matches = [m for m in matching if (m[1] != -1) && (m[2] != -1)]

	#println(length(unmatched_matches))

	cur_seq_sim = seq_similarity(U,V,matched_matches)
	#cur_top_sim, cur_motif_matched = top_similarity(B_motifs, A_ten,V,U,matching)
	if typeof(B_ten) === ThirdOrderSymTensor
		mode = 1
	else
		mode = 2
	end
	cur_motif_matched = TAME_score(B_motifs,A_ten,Set(size(A_ten.indices,mode):-1:1), matching)
		#TODO: check if this can be done with a TAME_score function

	#is it faster to precomputer X = UV'?
	unprocessed_matches = sort(matched_matches,by=match->dot(U[match[1],:],V[match[2],:]),rev=true)

	prev_matching = copy(matching)
	updatable_matching = copy(matching)
	updatable_inverted_matching = Dict([(j,i) for (i,j) in updatable_matching])

	for (i,_) in unprocessed_matches

		ip = get(updatable_matching,i,-1)

		if ip != -1 
			pref_B = Set(vcat(knn(T_V, V[ip,:], minimum((k,n)))[1],findnz(B[:,ip])[1]))
		else
			pref_B = Set()
		end
		
		if i != -1
			pref_A = Set(vcat(knn(T_U, U[i,:], minimum((k,m)))[1],findnz(A[:,i])[1])) 
																		# assuming symmetric matrices
			start_A_edge_mask = Set(A_node_edge_indices[i])
		else
			pref_A = Set()
			start_A_edge_mask = Set()
		end
	

		#
		#  check the local neighborhood of ip in B
		# 

		for jp in pref_B 

	
			j = get(updatable_inverted_matching,jp,-1)

			# -- swap edges + compute change in  sequence scores -- #
			updatable_matching[i] = jp
			if j != -1 #if jp was matched 

				updatable_matching[j] = ip
				A_edge_mask = union(start_A_edge_mask,Set(A_node_edge_indices[j]))
				
				# -- Compute sequence sim change -- # 

				if ip != -1
					start_seq_sim,t = @timed dot(U[i,:],V[ip,:]) + dot(U[j,:],V[jp,:])
					seq_scoring_runtimes += t
					end_seq_sim,t =@timed dot(U[i,:],V[jp,:]) + dot(U[j,:],V[ip,:])
					seq_scoring_runtimes += t
				else
					start_seq_sim,t = @timed dot(U[j,:],V[jp,:])
					seq_scoring_runtimes += t
					end_seq_sim,t =@timed dot(U[i,:],V[jp,:])
					seq_scoring_runtimes += t
				end
				# doing it in loop means vectors 
				# are more likely to be cached

			else

				if ip == -1 
					continue # both are unmatched, 
							 # no neighborhood to check in B
				end

				A_edge_mask = start_A_edge_mask

				start_seq_sim,t = @timed dot(U[i,:],V[ip,:])
				seq_scoring_runtimes += t
				end_seq_sim,t = @timed dot(U[i,:],V[jp,:])
				seq_scoring_runtimes += t

			end

			seq_sim_change = end_seq_sim - start_seq_sim

			#
			# Compute Change in topogical similarity
			#


			#(start_top_sim,start_motif_count),t = @timed top_similarity(B_motifs,A_ten,A_edge_mask,V,U,prev_matching)
			(start_motif_count),t = @timed TAME_score(B_motifs,A_ten,A_edge_mask,prev_matching)
			
			top_scoring_runtimes += t

			#(end_top_sim,end_motif_count),t = @timed top_similarity(B_motifs,A_ten,A_edge_mask,V,U,updatable_matching)
			(end_motif_count),t = @timed TAME_score(B_motifs,A_ten,A_edge_mask,updatable_matching)
			
			top_scoring_runtimes += t
			motif_match_change = (end_motif_count - start_motif_count)
			#top_sim_change = (end_top_sim - start_top_sim)		
			
			#=
			if match_method == "top_sim"
				check_val = top_sim_change
			elseif match_method == "motif_count"
				check_val = motif_match_change
			end
			=#
			check_val = motif_match_change

			if check_val > 0 ||  (check_val == 0.0 && seq_sim_change > 0)

				cur_motif_matched += motif_match_change
				#cur_top_sim += top_sim_change
				cur_seq_sim += seq_sim_change

	
				# update the previous matching to be the same as cur
				prev_matching[i] = jp 
				prev_matching[j] = ip 

				# update invertible matchings				
				updatable_inverted_matching[ip] = j
				updatable_inverted_matching[jp] = i
							#we can do this at the end bc 
							#we only use them at the start
				
				ip = jp #update for future loops

			else  #change the matching back 

				updatable_matching[i] = ip
				updatable_matching[j] = jp

			end



		end

		#
		#  check the local neighborhood of i in A
		# 


		for j in pref_A 
	
			jp = get(updatable_matching,j,-1)

			updatable_matching[j] = ip
			updatable_matching[i] = jp

			if jp != -1

				A_edge_mask = union(start_A_edge_mask,Set(A_node_edge_indices[j]))
				
				if ip != -1
					start_seq_sim,t = @timed dot(U[i,:],V[ip,:]) + dot(U[j,:],V[jp,:])
					seq_scoring_runtimes += t
					end_seq_sim,t =@timed dot(U[i,:],V[jp,:]) + dot(U[j,:],V[ip,:])
					seq_scoring_runtimes += t
				else
					start_seq_sim,t = @timed dot(U[j,:],V[jp,:])
					seq_scoring_runtimes += t
					end_seq_sim,t =@timed dot(U[i,:],V[jp,:])
					seq_scoring_runtimes += t
				end

			else
				if ip == -1 
					continue # both are unmatched, 
							 # no neighborhood to check in B
				end

				A_edge_mask = start_A_edge_mask

				start_seq_sim,t = @timed dot(U[i,:],V[ip,:])
				seq_scoring_runtimes += t
				end_seq_sim,t = @timed dot(U[j,:],V[ip,:])
				seq_scoring_runtimes += t
			end
			seq_sim_change = end_seq_sim - start_seq_sim

			#
			# Compute Change in topogical similarity
			#
		
			#(start_top_sim,start_motif_count),t = @timed top_similarity(B_motifs,A_ten,A_edge_mask,V,U,prev_matching)
			(start_motif_count),t = @timed TAME_score(B_motifs,A_ten,A_edge_mask,prev_matching)
			top_scoring_runtimes += t

			#(end_top_sim, end_motif_count),t =   @timed top_similarity(B_motifs,A_ten,A_edge_mask,V,U,updatable_matching)
			(end_motif_count),t =   @timed TAME_score(B_motifs,A_ten,A_edge_mask,updatable_matching)
			
			top_scoring_runtimes += t
			motif_match_change = (end_motif_count - start_motif_count)
			#top_sim_change = (end_top_sim - start_top_sim)
			
			#=
			if match_method == "top_sim"
				check_val = top_sim_change
			elseif match_method == "motif_count"
				check_val = motif_match_change 
			end
			=#

			check_val = motif_match_change 

			if check_val > 0 ||  (check_val == 0.0 && seq_sim_change > 0)

				cur_motif_matched += motif_match_change
				#cur_top_sim += top_sim_change
				cur_seq_sim += seq_sim_change

				# update the previous matching to be the same as cur
				prev_matching[i] = jp 
				prev_matching[j] = ip 

				updatable_inverted_matching[ip] = j
				updatable_inverted_matching[jp] = i

				ip = jp 
				#matching = copy(updatable_matching)

			else #change back the matching
				updatable_matching[i] = ip
				updatable_matching[j] = jp
			end
		end

	end

	return updatable_matching, seq_scoring_runtimes, top_scoring_runtimes

end


function tabu_search(A,B,A_ten,B_ten,U,V,matching,k)#,match_method="top_sim")


	m = size(A,1)
	n = size(B,1)

	T_U = BallTree(U')
	T_V = BallTree(V')

	B_motifs = Set{Vector{Int}}()

	if typeof(B_ten) == ThirdOrderSymTensor
		for idx = 1:size(B_ten.indices,1)
			push!(B_motifs,B_ten.indices[idx,:])
		end
	else
		for idx = 1:size(B_ten.indices,2)
			push!(B_motifs,B_ten.indices[:,idx])
		end
	end

	A_node_edge_indices = create_edge_idx_map(A_ten)

	#seq_scoring_runtimes = 0.0
	#top_scoring_runtimes = 0.0

	matched_matches = [m for m in matching if (m[1] != -1) && (m[2] != -1)]

	#println(length(unmatched_matches))

	cur_seq_sim = seq_similarity(U,V,matched_matches)

	if typeof(B_ten) === ThirdOrderSymTensor
		mode = 1
	else
		mode = 2
	end
	cur_motif_matched = TAME_score(B_motifs, A_ten, Set(size(A_ten.indices,mode):-1:1),matching)
													#matching doesn't reorient using this orientation
	
	#is it faster to precomputer X = UV'?
	unprocessed_matches = sort(matched_matches,by=match->dot(U[match[1],:],V[match[2],:]),rev=true)

	prev_matching = copy(matching)
	updatable_matching = copy(matching)
	updatable_inverted_matching = Dict([(j,i) for (i,j) in updatable_matching])

	for (i,_) in unprocessed_matches

		ip = get(updatable_matching,i,-1)

		if ip != -1 
			pref_B = Set(vcat(knn(T_V, V[ip,:], minimum((k,n)))[1],findnz(B[ip,:])[1]))
			#append!(pref_B,sample(sample(1:B.n,10)))
		else
			pref_B = Set()
		end
		
		if i != -1
			pref_A = Set(vcat(knn(T_U, U[i,:], minimum((k,m)))[1],findnz(A[:,i])[1])) 
																		# assuming symmetric matrices
			#append!(pref_A,sample(sample(1:A.n,10)))
			start_A_edge_mask = Set(A_node_edge_indices[i])
		else
			pref_A = Set()
			start_A_edge_mask = Set()
		end
	

		#
		#  check the local neighborhood of ip in B
		# 

		for jp in pref_B 

	
			j = get(updatable_inverted_matching,jp,-1)

			# -- swap edges + compute change in  sequence scores -- #
			updatable_matching[i] = jp
			if j != -1 #if jp was matched 

				updatable_matching[j] = ip
				A_edge_mask = union(start_A_edge_mask,Set(A_node_edge_indices[j]))
				
				# -- Compute sequence sim change -- # 
				if ip != -1
					start_seq_sim = dot(U[i,:],V[ip,:]) + dot(U[j,:],V[jp,:])
					end_seq_sim = dot(U[j,:],V[ip,:]) + dot(U[i,:],V[jp,:])
				else
					start_seq_sim = dot(U[j,:],V[jp,:])
					end_seq_sim = dot(U[i,:],V[jp,:])
				end
				
							# doing it in loop means vectors 
							# are more likely to be cached

			else

				if ip == -1 
					continue # both are unmatched, 
							 # no neighborhood to check in B
				end
				A_edge_mask = start_A_edge_mask

				start_seq_sim = dot(U[i,:],V[ip,:])
				end_seq_sim = dot(U[i,:],V[jp,:])

			end

			seq_sim_change = end_seq_sim - start_seq_sim

			#
			# Compute Change in topogical similarity
			#


			start_motif_count = TAME_score(B_motifs,A_ten,A_edge_mask,prev_matching)
			end_motif_count = TAME_score(B_motifs,A_ten,A_edge_mask,updatable_matching)
		
			motif_match_change = (end_motif_count - start_motif_count)
			
			#=
			if match_method == "top_sim"
				check_val = top_sim_change
			elseif match_method == "motif_count"
				check_val = motif_match_change
			end
			=#
			check_val = motif_match_change

			if check_val > 0 ||  (check_val == 0.0 && seq_sim_change > 0)

				cur_motif_matched += motif_match_change
				#cur_top_sim += top_sim_change
				cur_seq_sim += seq_sim_change

	
				# update the previous matching to be the same as cur
				prev_matching[i] = jp 
				prev_matching[j] = ip 

				# update invertible matchings				
				updatable_inverted_matching[ip] = j
				updatable_inverted_matching[jp] = i
							#we can do this at the end bc 
							#we only use them at the start
				
				ip = jp #update for future loops

			else  #change the matching back 

				updatable_matching[i] = ip
				updatable_matching[j] = jp

			end



		end

		#
		#  check the local neighborhood of i in A
		# 


		for j in pref_A 
	
			jp = get(updatable_matching,j,-1)

			updatable_matching[j] = ip
			updatable_matching[i] = jp

			if jp != -1

				A_edge_mask = union(start_A_edge_mask,Set(A_node_edge_indices[j]))
				
				#start_seq_sim,t = @timed dot(U[i,:],V[ip,:]) + dot(U[j,:],V[jp,:])
				#seq_scoring_runtimes += t

				if ip != -1
					start_seq_sim = dot(U[i,:],V[ip,:]) + dot(U[j,:],V[jp,:])
					end_seq_sim = dot(U[j,:],V[ip,:]) + dot(U[i,:],V[jp,:])
				else
					start_seq_sim = dot(U[j,:],V[jp,:])
					end_seq_sim = dot(U[i,:],V[jp,:])
				end

				#seq_scoring_runtimes += t

			else

				if ip == -1 
					continue # both are unmatched, 
							 # no neighborhood to check in B
				end
				A_edge_mask = start_A_edge_mask

				start_seq_sim = dot(U[i,:],V[ip,:])
				end_seq_sim = dot(U[j,:],V[ip,:])

			end
			seq_sim_change = end_seq_sim - start_seq_sim

			#
			# Compute Change in topogical similarity
			#
		
			#=
			(start_top_sim,start_motif_count),t = @timed top_similarity(B_motifs,A_ten,A_edge_mask,V,U,prev_matching)
			top_scoring_runtimes += t

			(end_top_sim, end_motif_count),t =   @timed top_similarity(B_motifs,A_ten,A_edge_mask,V,U,updatable_matching)
			top_scoring_runtimes += t
			=#

			start_motif_count = TAME_score(B_motifs,A_ten,A_edge_mask,prev_matching)
			end_motif_count = TAME_score(B_motifs,A_ten,A_edge_mask,updatable_matching)
		
			motif_match_change = (end_motif_count - start_motif_count)
			#top_sim_change = (end_top_sim - start_top_sim)
			
			#=
			if match_method == "top_sim"
				check_val = top_sim_change
			elseif match_method == "motif_count"
				check_val = motif_match_change 
			end
			=#

			check_val = motif_match_change 

			if check_val > 0 ||  (check_val == 0.0 && seq_sim_change > 0)

				cur_motif_matched += motif_match_change
				#cur_top_sim += top_sim_change
				cur_seq_sim += seq_sim_change

				# update the previous matching to be the same as cur
				prev_matching[i] = jp 
				prev_matching[j] = ip 

				updatable_inverted_matching[ip] = j
				updatable_inverted_matching[jp] = i

				ip = jp 
				#matching = copy(updatable_matching)

			else #change back the matching
				updatable_matching[i] = ip
				updatable_matching[j] = jp
			end
		end

	end

	return updatable_matching

end

