abstract type PostProcessingMethod end

struct KlauAlgo <: PostProcessingMethod 
    k::Int
end
KlauAlgo() = KlauAlgo(-1)
struct noPostProcessing <: PostProcessingMethod end

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
                    U::Matrix{T},V::Matrix{T};kwargs...) where T
    (_,_,matching,_)=  bipartite_matching_primal_dual(U*V';kwargs...)
    return netalignmr(A,B,knearest_sparsification(U,V,matching_array_to_dict(matching),2*min(size(U,2),size(V,2))))
end

function netalignmr(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
                    U::Matrix{T},V::Matrix{T},k::Int;kwargs...) where T
    (_,_,matching,_)=  bipartite_matching_primal_dual(U*V';kwargs...)
    return netalignmr(A,B,knearest_sparsification(U,V,matching_array_to_dict(matching),k))
end

function netalignmr(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
                    U::Matrix{T},V::Matrix{T},
                    matching::Union{Vector{NTuple{2,Int}},Dict{Int,Int}},k::Int) where T
    return netalignmr(A,B,knearest_sparsification(U,V,matching,k))
end

function netalignmr(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int},
                    U::Matrix{T},V::Matrix{T},
                    matching::Union{Vector{NTuple{2,Int}},Dict{Int,Int}}) where T
    return netalignmr(A,B,knearest_sparsification(U,V,matching,2*min(size(U,2),size(V,2))))
end


function netalignmr(A::SparseMatrixCSC{T1,Int},B::SparseMatrixCSC{T1,Int},L::SparseMatrixCSC{T2,Int}) where {T1,T2}

	@assert size(L,1) == size(A,1) && size(L,2) == size(B,1)

	#L = sparse(X)
	S,w,li,lj = netalign_setup(A,B,L)

	# align networks
	a = 1;
	b = 1;
	stepm = 25;
	rtype = 1;
	maxiter = 10;
	verbose = true;
	gamma = 0.4;
	(xbest,st,status,hist), t_netalignmr = @timed NetworkAlign.netalignmr(S,w,a,b,li,lj,gamma,stepm,rtype,maxiter,verbose)

    #=
    matching = Dict{Int,Int}()
	for (x,i,j) in zip(xbest,li,lj)
		if x == 1.0
            matching[j] = i
		end
	end
    =#
    matching = matching_array_to_dict(bipartite_matching(sparse(li,lj,xbest)).match)

	return matching, t_netalignmr, L

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