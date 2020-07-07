#=
PostProcessing:
- Julia version: 
- Author: dgleich
- Date: 2020-05-22
=#

#Notes: change b-matching search order to k=1:n, i = 1:n , j = k+1 - i (verify)

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
function b_matching(u::Array{Float64,1}, v::Array{Float64,1}, b::Int)

    @assert length(u) <= length(v) #make first vector shorter

    n = length(u)

    #get indices sorted by increasing order
    u_indices = sort(1:length(u),by=i->-u[i])
    v_indices = sort(1:length(v),by=i->-v[i])

    matching_degrees = zeros(length(v))

    Matchings = Array{Tuple{Int,Int,Float64},1}(undef,b*n)  #-1 indicates unset matching
    match_idx = 1
    starting_index = 0b1
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