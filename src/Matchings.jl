#TODO: update return_timings=false to a TypeFlag()
abstract type MatchingFlag end
struct returnTimings <: MatchingFlag end
struct noTimings <: MatchingFlag end


#match1 gets priority over match2
function combine_matchings(match1::Dict{Int,Int},match2::Dict{Int,Int})

    new_matching = copy(match1)
    for (i,j) in match2
        if !haskey(new_matching,i)
            new_matching[i] = j
        end
    end
    return new_matching
end

function degree_based_matching(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int}) where T

	m = size(A,1)
	n = size(B,1)

    noise_factor = .01
	d_A = A*ones(m) .+ noise_factor*rand(m)
	d_B = B*ones(n) .+ noise_factor*rand(n)

	
	p_A = sort(1:m,by=i->d_A[i]) #breaks ties with some variation
	p_B = sort(1:n,by=j->d_B[j])

	return p_A,p_B

end

function low_rank_matching(U::Array{T,2},V::Array{T,2}) where T
    n,d1 = size(U)
    m,d2 = size(V)
    @assert d1 == d2

    matching_weights = zeros(T,d1)
    matchings = Array{Dict{Int,Int},1}(undef,d1)

    for i = 1:d1
        matchings[i],matching_weights[i] = rank_one_matching(U[:,i],V[:,i])
    end


    D = zeros(d1,d2)
    for j in 1:d2
        for i in 1:d1
            w = 0.0
            for (v_i,v_j) in matchings[j]
                edge_w = U[v_i,i]*V[v_j,i]
                if edge_w > 0
                    w += edge_w
                end
            end
            D[i,j] = matching_weights[i]/w
        end
    end

    d_j = maximum(D,dims= 1)
    opt_j = argmin(d_j).I[2]

    return matchings[opt_j], matching_weights[opt_j]

end


function rank_one_matching(u::Array{T,1},v::Array{T,1}) where T

    #get maximum matching by rearrangement theorem
    u_perm = sortperm(u,rev=true)
    v_perm = sortperm(v,rev=true)

    matching_weight = zero(T )
    Match_mapping = Dict{Int,Int}()
    for (i,j) in zip(u_perm,v_perm)

        #only match positive to positive and negative to negative
        if (u[i] > 0 && v[j] > 0 ) || (u[i] < 0 && v[j] < 0)
            Match_mapping[i] = j
            matching_weight += u[i]*v[j]
        end
    end
    return Match_mapping, matching_weight

end

function search_Krylov_space(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,U::Array{Float64,2},V::Array{Float64,2};returnScoringTimings::T=noTimings()) where {T <:MatchingFlag}

    best_score = -1
    best_i = -1
    best_j = -1
    best_matching = Dict{Int,Int}()

    Triangle_check = Dict{Array{Int,1},Int}()
    A_unique_nnz = length(A.values)
    B_unique_nnz = length(B.values)

    if A_unique_nnz > B_unique_nnz
        for i in 1:A_unique_nnz
            Triangle_check[A.indices[:,i]] = 1
        end
        Input_tensor = B
    else
        for i in 1:B_unique_nnz
            Triangle_check[B.indices[:,i]] = 1
        end
        Input_tensor = A
    end

    scoring_time =0.0
    for i in 1:size(U,2)
       for j in 1:size(V,2)
            A_to_B_mapping, _ = rank_one_matching(U[:,i],V[:,j])
            B_to_A_mapping, _ = rank_one_matching(V[:,j],U[:,i])
            if typeof(returnScoringTimings) === returnTimings
                if A_unique_nnz > B_unique_nnz
                    (matched_tris,gaped_tris),t = @timed TAME_score(Triangle_check,Input_tensor,B_to_A_mapping)
                else
                    (matched_tris,gaped_tris),t = @timed TAME_score(Triangle_check,Input_tensor,A_to_B_mapping)
                end
                scoring_time += t
            else # noTimings
                if A_unique_nnz > B_unique_nnz
                    matched_tris,gaped_tris = TAME_score(Triangle_check,Input_tensor,B_to_A_mapping)
                else
                    matched_tris,gaped_tris = TAME_score(Triangle_check,Input_tensor,A_to_B_mapping)
                end
            end

            if matched_tris > best_score
                best_matching = A_to_B_mapping
                best_score = matched_tris
                best_i = i
                best_j = j
            end
        end
    end
    if typeof(returnScoringTimings) === returnTimings
        return best_score, best_i, best_j, best_matching, scoring_time
    else
        return best_score, best_i, best_j, best_matching
    end
end

function search_Krylov_space(A::SymTensorUnweighted{S},B::SymTensorUnweighted{S},U::Array{Float64,2},V::Array{Float64,2};returnScoringTimings::T=noTimings())where {S <: Motif,T <:MatchingFlag}


    A_order,A_unique_nnz = size(A.indices)
    B_order,B_unique_nnz = size(B.indices)
    #@assert A_order == B_order

    best_score = -1
    best_i = -1
    best_j = -1
    best_matching = Dict{Int,Int}()

    motif_check = Dict{Array{Int,1},Int}()


    if A_unique_nnz > B_unique_nnz
        for i in 1:A_unique_nnz
            motif_check[A.indices[:,i]] = 1
        end
        Input_tensor = B
    else
        for i in 1:B_unique_nnz
            motif_check[B.indices[:,i]] = 1
        end
        Input_tensor = A
    end
    
    scoring_time =0.0
    for i in 1:size(U,2)
       for j in 1:size(V,2)
            A_to_B_mapping, _ = rank_one_matching(U[:,i],V[:,j])
            B_to_A_mapping, _ = rank_one_matching(V[:,j],U[:,i])
            if typeof(returnScoringTimings) === returnTimings
                if A_unique_nnz > B_unique_nnz
                    (matched_motifs,gaped_motifs),t = @timed TAME_score(motif_check,Input_tensor,B_to_A_mapping)
                else
                    (matched_motifs,gaped_motifs),t = @timed TAME_score(motif_check,Input_tensor,A_to_B_mapping)
                end
                scoring_time += t
            else
                if A_unique_nnz > B_unique_nnz
                    matched_motifs,gaped_motifs = TAME_score(motif_check,Input_tensor,B_to_A_mapping)
                else
                    matched_motifs,gaped_motifs = TAME_score(motif_check,Input_tensor,A_to_B_mapping)
                end
            end

            if matched_motifs > best_score
                best_matching = A_to_B_mapping
                best_score = matched_motifs
                best_i = i
                best_j = j
            end
        end
    end
    if typeof(returnScoringTimings) === returnTimings
        return best_score, best_i, best_j, best_matching, scoring_time
    else
        return best_score, best_i, best_j, best_matching   
    end
end

#=
function search_Krylov_space(A::Array{SymTensorUnweighted,1},B::Array{SymTensorUnweighted,1},U::Array{Float64,2},V::Array{Float64,2})


    A_order,A_unique_nnz = size(A.indices)
    B_order,B_unique_nnz = size(B.indices)
    #@assert A_order == B_order

    best_score = -1
    best_i = -1
    best_j = -1
    best_matching = Dict{Int,Int}()

    motif_check = Dict{Array{Int,1},Int}()


    if A_unique_nnz > B_unique_nnz
        for i in 1:A_unique_nnz
            motif_check[A.indices[:,i]] = 1
        end
        Input_tensor = B
    else
        for i in 1:B_unique_nnz
            motif_check[B.indices[:,i]] = 1
        end
        Input_tensor = A
    end

    for i in 1:size(U,2)
       for j in 1:size(V,2)

            if A_unique_nnz > B_unique_nnz
                matched_motifs,gaped_motifs,matching = TAME_score(motif_check,Input_tensor,V[:,j],U[:,i])
            else
                matched_motifs,gaped_motifs,matching = TAME_score(motif_check,Input_tensor,U[:,i],V[:,j])
            end

            if matched_motifs > best_score
                best_matching = matching
                best_score = matched_motifs
                best_i = i
                best_j = j
            end
        end
    end
    return best_score, best_i, best_j, best_matching
end
=#

function search_Krylov_space(A::Array{SymTensorUnweighted{S},1},B::Array{SymTensorUnweighted{S},1},U::Array{Float64,2},V::Array{Float64,2};returnScoringTimings::T=noTimings()) where {S <: Motif,T <:MatchingFlag}

    @assert length(A) == length(B)
    for i = 1:length(A) #ensure orders are the same at each i
        @assert size(A[i].indices,1) == size(B[i].indices,1)
    end
 
    #create sets of edges
    larger_edge_sets = [ ]
    
    for (A_motifs,B_motifs) in zip(A,B)
        if size(A_motifs.indices,2) > size(B_motifs.indices,2)  #larger set of edges gets 
            #push!(larger_edge_sets,Set(eachcol(A_motifs.indices)))
            push!(larger_edge_sets,Dict([(Array(col),1) for col in eachcol(A_motifs.indices)]))
        else
            push!(larger_edge_sets,Dict([(Array(col),1) for col in eachcol(B_motifs.indices)]))
        end
    end

    best_matching_score = -1
    best_matching_idx = (-1,-1)
    best_matched_motifs::Array{Int,1} = []
    best_mapping::Dict{Int,Int} = Dict()

    max_iter = size(U,2) - 1 #skip the first column of U and V
    scoring_timing = 0.0


    for i=1:max_iter
        for j=1:max_iter
            A_to_B_mapping, _ = rank_one_matching(U[:,i],V[:,j])
            B_to_A_mapping, _ = rank_one_matching(V[:,j],U[:,i])

            
            #TODO: handle mapping inversion 

            #matched_motifs, matching_score = motif_matching_counts(A,B,mapping)
            matched_motifs = zeros(Int,length(A))
            if typeof(returnScoringTimings) === returnTimings
                
                for i =1:length(A)
                    if size(A[i].indices,2) > size(B[i].indices,2)
                        (matched_motifs[i],_),t = @timed TAME_score(larger_edge_sets[i], B[i], B_to_A_mapping)
                    else
                        (matched_motifs[i],_),t = @timed TAME_score(larger_edge_sets[i], A[i], A_to_B_mapping)
                    end
                    scoring_timing += t
                end
            else
                for i =1:length(A)
                    if size(A[i].indices,2) > size(B[i].indices,2)
                        matched_motifs[i],_ = TAME_score(larger_edge_sets[i], B[i], B_to_A_mapping)
                    else
                        matched_motifs[i],_ = TAME_score(larger_edge_sets[i], A[i], A_to_B_mapping)
                    end
                end
            end
            #matching_score = motif_matching_count(B[end], A[end], mapping)
            
            matching_score = 0 
            for i= 1:length(A)
                matching_score += binomial(size(A[i].indices,1),2)*matched_motifs[i]
            end

            if matching_score > best_matching_score
                best_matching_score = matching_score
                best_matching_idx = (i,j)
                best_matched_motifs = copy(matched_motifs)
                best_mapping = A_to_B_mapping
            end
        
        end
    end

    best_i, best_j = best_matching_idx
    if typeof(returnScoringTimings) === returnTimings
        return best_matching_score, best_matched_motifs,  best_i, best_j, best_mapping, scoring_timing
    else
        return best_matching_score, best_matched_motifs,  best_i, best_j, best_mapping
    end
end

#TODO: TEST
#used when we don't want to recreate the triangle matching dictionary multiple times
function TAME_score(Triangle_Dict::Dict{Array{Int,1},Int},Input_tensor,
                    u::Array{Float64,1},v::Array{Float64,1})

    Match_mapping, _ = rank_one_matching(u,v)
    return TAME_score(Triangle_Dict,Input_tensor,Match_mapping)...,Match_mapping

end



function TAME_score(A,B,U::Array{Float64,2},V::Array{Float64,2};return_timings::T=noTimings()) where {T <: MatchingFlag}

    if typeof(return_timings) === returnTimings
        (Match_mapping, _), matching_time = @timed low_rank_matching(U,V)
        (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,Match_mapping)
        return triangle_count, gaped_triangles,Match_mapping, matching_time, scoring_time 
    else
        Match_mapping,weight = low_rank_matching(U,V)
        return TAME_score(A,B,Match_mapping)...,Match_mapping
    end

end

function TAME_score(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor, X::SparseMatrixCSC{Float64,Int64};return_timings::T=noTimings()) where {T <: MatchingFlag}

    if typeof(return_timings) === returnTimings
        x ,bipartite_matching_time = @timed bipartite_matching(X)
        matching_dict = Dict(enumerate(x.match))
        (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,matching_dict)
        return triangle_count, gaped_triangles, matching_dict, bipartite_matching_time, scoring_time
    else
        x = bipartite_matching(X)
        matching_dict = Dict(enumerate(x.match))
        return TAME_score(A,B,matching_dict)..., matching_dict
    end

end

function TAME_score(A::Union{ThirdOrderSymTensor,Array{SymTensorUnweighted{S},1}},
                    B::Union{ThirdOrderSymTensor,SymTensorUnweighted{S},Array{SymTensorUnweighted{S},1}},X::Array{Float64,2};return_timings::T=noTimings(),kwargs...) where {T <: MatchingFlag,S <: Motif}

    if typeof(return_timings) === returnTimings
        (_,_,matching,_) ,matching_time = @timed bipartite_matching_primal_dual(X;kwargs...)
        matching_dict = matching_array_to_dict(matching)
        (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,matching_dict)
        return triangle_count, gaped_triangles, matching_dict, matching_time, scoring_time
    else
        _,_,matching,_ = bipartite_matching_primal_dual(X;kwargs...)
        matching_dict = matching_array_to_dict(matching)
        return TAME_score(A,B,matching_dict)...,matching_dict
    end
end

function TAME_score(A::SymTensorUnweighted{S},B::SymTensorUnweighted{S},X::Array{Float64,2};return_timings::T=noTimings(),kwargs...) where {T <: MatchingFlag,S <: Motif}

    if typeof(return_timings) === returnTimings
        (_,_,matching,_) ,matching_time = @timed bipartite_matching_primal_dual(X;kwargs...)
        matching_dict = matching_array_to_dict(matching)
        (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,matching_dict)
        return triangle_count, gaped_triangles, matching_dict, matching_time, scoring_time
    else
        _,_,matching,_ = bipartite_matching_primal_dual(X;kwargs...)
        matching_dict = matching_array_to_dict(matching)
        #return TAME_score(A,B,matching_array_to_dict(matching))
        return TAME_score(A,B,matching_dict)...,matching_dict
        #BUG?
    end
end


function TAME_score(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,x::Array{Float64,1};return_timings::T=noTimings(),kwargs...) where {T <: MatchingFlag}

    X = reshape(x,A.n,B.n)
    if typeof(return_timings) === returnTimings
        (_,_,matching,_) ,matching_time = @timed bipartite_matching_primal_dual(X;kwargs...)
        matching_dict = matching_array_to_dict(matching)
        (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,matching_dict)
        return triangle_count, gaped_triangles, matching_dict, matching_time, scoring_time
    else
        _,_,matching,_ = bipartite_matching_primal_dual(X;kwargs...)
        matching_dict = matching_array_to_dict(matching)
        return TAME_score(A,B,matching_dict)...,matching_dict
    end

end



"""
   Match_mapping is expected to map V_A -> V_B
"""

function TAME_score(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,Match_mapping::Dict{Int,Int})
    # matching oriented A -> B 


    match_len = length(Match_mapping)

    Triangle_check = Dict{Array{Int,1},Int}()
    gaped_triangles = 0
    triangle_count = 0

    if size(A.indices,2) > size(B.indices,2)

        for i in 1:size(A.indices,2)
            Triangle_check[A.indices[:,i]] = 1
        end

        #invert to map V_B indices to V_A
        Match_mapping = Dict(value => key for (key, value) in Match_mapping)

        for i in 1:size(B.indices,2)
            v_i,v_j,v_k = B.indices[:,i]

            matched_triangle =
              sort([get(Match_mapping,v_i,-1),get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])
            match = get(Triangle_check,matched_triangle,0)
            if match == 1
                triangle_count += 1
            else
                gaped_triangles += 1
            end
        end

    else
        for i in 1:size(B.indices,2)
            Triangle_check[B.indices[:,i]] = 1
        end

        for i in 1:size(A.indices,2)
            v_i,v_j,v_k = A.indices[:,i]
            matched_triangle =
               sort([get(Match_mapping,v_i,-1), get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])

            match = get(Triangle_check,matched_triangle,0)
            if match == 1
                triangle_count += 1
            else
                gaped_triangles += 1
            end
        end
    end

   # triangles, triangle_weight = count_triangles(sub_A,sub_B)

    return triangle_count, gaped_triangles

end

#TODO: may remove the array mapping routines 
function TAME_score(A::SymTensorUnweighted{S},B::SymTensorUnweighted{S}, mapping::Array{Int,1}) where {S <: Motif}

    # matching oriented B -> A
    @assert size(A.indices,1) == size(B.indices,1)

    if size(B.indices,2) > size(A.indices,2)
        inverted_mapping = -ones(Int, maximum(mapping))

        for (v,u) in enumerate(mapping)
            if u != -1
                inverted_mapping[u] = v
            end
        end

        #return (-1,"ERROR:mapping needs to be inverted")
        return TAME_score(B, A,inverted_mapping)
    end

    order =  size(A.indices,1)
    A_motifs = Set(eachcol(A.indices))

    matched_motifs = 0
    missed_motifs = 0

    for idx =1:size(B.indices,2)
        edge = B.indices[:,idx]
        new_edge = [mapping[i] for i in edge]
        sort!(new_edge)

        if new_edge in A_motifs
            matched_motifs += 1
        else 
            missed_motifs += 1
        end

    end
    return matched_motifs, matched_motifs

end

function TAME_score(A::Array{SymTensorUnweighted{S},1},B::Array{SymTensorUnweighted{S},1},A_to_B_mapping::Dict{Int,Int}) where {S <: Motif}
    # matching oriented A -> B 

    @assert length(A) == length(B)
    for i = 1:length(A) #ensure orders are the same at each i
        @assert size(A[i].indices,1) == size(B[i].indices,1)
    end

    B_to_A_mapping::Dict{Int,Int} = Dict([(j,i) for (i,j) in A_to_B_mapping])

    #create sets of edges
    larger_edge_sets = [ ]
    
    for (A_motifs,B_motifs) in zip(A,B)
        if size(A_motifs.indices,2) > size(B_motifs.indices,2)  #larger set of edges gets 
            #push!(larger_edge_sets,Set(eachcol(A_motifs.indices)))
            push!(larger_edge_sets,Dict([(Array(col),1) for col in eachcol(A_motifs.indices)]))
        else
            push!(larger_edge_sets,Dict([(Array(col),1) for col in eachcol(B_motifs.indices)]))
        end
    end

    best_matching_score = -1
    best_matched_motifs::Array{Int,1} = []

    matched_motifs = zeros(Int,length(A))
    for i =1:length(A)
        if size(A[i].indices,2) > size(B[i].indices,2)
            matched_motifs[i],_ = TAME_score(larger_edge_sets[i], B[i], B_to_A_mapping)
        else
            matched_motifs[i],_ = TAME_score(larger_edge_sets[i], A[i], A_to_B_mapping)
        end
    end
    #matching_score = motif_matching_count(B[end], A[end], mapping)
    
    matching_score = 0 
    for i= 1:length(A)
        #println(binomial(size(A[i],1),2))
        #println(matched_motifs[i])
        matching_score += binomial(size(A[i].indices,1),2)*matched_motifs[i]
    end

    return matching_score, matched_motifs
end

function TAME_score(A::SymTensorUnweighted{Clique},B::SymTensorUnweighted{Clique}, mapping::Dict{Int,Int})
    # matching oriented A -> B 

    @assert size(A.indices,1) == size(B.indices,1)

    #=
    if size(B.indices,2) > size(A.indices,2)
        println("inverting")
        inverted_mapping = Dict{Int,Int}()

        for (v,u) in mapping
            if u != -1
                inverted_mapping[u] = v
            end
        end

        #return (-1,"ERROR:mapping needs to be inverted")
        return TAME_score(B, A,inverted_mapping)
    end
    =#

    order =  size(A.indices,1)
    B_motifs = Set(eachcol(B.indices))

    matched_motifs = 0
    gaped_motifs = 0

    for idx =1:size(A.indices,2)
        edge = A.indices[:,idx]
        new_edge = sort([get(mapping,i,-1) for i in edge])
        if new_edge in B_motifs
            matched_motifs += 1
        else
            gaped_motifs += 1
        end

    end
    return matched_motifs, gaped_motifs

end

function TAME_score(A::SymTensorUnweighted{Cycle},B::SymTensorUnweighted{Cycle}, mapping::Dict{Int,Int})
    # matching oriented B -> A 
    
    @assert size(A.indices,1) == size(B.indices,1)

    if size(B.indices,2) > size(A.indices,2)
        inverted_mapping = Dict{Int,Int}()

        for (v,u) in mapping
            if u != -1
                inverted_mapping[u] = v
            end
        end
        return TAME_score(B, A,inverted_mapping)
    end

    order =  size(A.indices,1)
    A_cycle_hashes = Set{Float64}()
	for i= 1:size(A.indices,2)
		#A.indices are assumed to have the smallest index a the start of the column
		push!(A_cycle_hashes,DistributedTensorConstruction.cycle_hash(A.indices[:,i]))
	end

    matched_motifs = 0
    missed_motifs = 0

	l = size(B.indices,1)
    for idx =1:size(B.indices,2)
        edge = B.indices[:,idx]
        new_edge = [get(mapping,i,-1) for i in edge]

        new_edge = [new_edge[(i+argmin(new_edge)+2)%l+1] for i in 1:l]
			#cycles the edge so smallest index is first
		hash = DistributedTensorConstruction.cycle_hash(new_edge)	
	
        if hash in A_cycle_hashes
            matched_motifs += 1
        else
            missed_motifs += 1
        end

    end

    return matched_motifs, missed_motifs
end



function TAME_score(Triangle_Dict::Dict{Array{Int,1},Int},Input_tensor::ThirdOrderSymTensor,
                    Match_mapping::Union{Dict{Int,Int},Vector{Int}})

    triangle_count = 0
    gaped_triangles = 0

    for i in 1:length(Input_tensor.values)
        v_i,v_j,v_k = Input_tensor.indices[:,i]

        matched_triangle =
          sort([get(Match_mapping,v_i,-1),get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])

        match = get(Triangle_Dict,matched_triangle,0)
        if match == 1
            triangle_count += 1
        else
            gaped_triangles += 1
        end
    end
    return triangle_count, gaped_triangles
end


#TODO: may remove, see above

function TAME_score(A_motifs::Dict{Array{Int,1},Int}, B::SymTensorUnweighted{S}, mapping::Array{Int,1}) where {S <: Motif}

  
    order =  size(B.indices,1)
    #A_motifs = Set(eachcol(Indices_A))

    matched_motifs = 0
    missed_motifs = 0

    for idx =1:size(B.indices,2)
        edge = B.indices[:,idx]
        new_edge = [mapping[i] for i in edge]
        sort!(new_edge)

        if new_edge in A_motifs
            matched_motifs += 1
        else
            missed_motifs += 1
        end

    end
    return matched_motifs, missed_motifs

end


function TAME_score(A_motifs::Dict{Array{Int,1},Int}, B::SymTensorUnweighted{S}, mapping::Dict{Int,Int}) where {S <: Motif}

  
    order =  size(B.indices,1)
    #A_motifs = Set(eachcol(Indices_A))

    matched_motifs = 0
    gaped_motifs = 0

    for idx =1:size(B.indices,2)
        edge = B.indices[:,idx]
        new_edge = [get(mapping,i,-1) for i in edge]
        sort!(new_edge)

        if haskey(A_motifs,new_edge)
            matched_motifs += 1
        else
            gaped_motifs += 1
        end

    end
    return matched_motifs, gaped_motifs

end

function edges_matched(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int}, mapping::Dict{Int,Int}) where T

    edges_matched = 0
    edges_missed = 0

    B_nnz = Set(zip(findnz(B)[1:2]...))
    
    for (i,j,_) in zip(findnz(A)...)

        ip = get(mapping,i,-1)
        jp = get(mapping,j,-1)

        if (ip,jp) in B_nnz
            edges_matched += 1
        else
            edges_missed += 1
        end

    end

    return edges_matched, edges_missed
end

function matching_array_to_dict(matching::Vector{Int})

    matching_dict = Dict{Int,Int}()
    for (i,j) in enumerate(matching)
        if i != -1 && j != -1 #-1 indicates unmatched
            matching_dict[i] = j
        end
    end
    return matching_dict
end



"""------------------------------------------------------------------------------
  A function for solving the a maximum bipartite matching problem with a dense 
  m x n input X. 

  Inputs
  ------
  * X - (Matrix{Float64}):
    The weights of the input bipartite graph. 
  * tol - (Float):
    The tolerance to determine whether or not the flow through an edge is at 
    capacity. 
  * 'normalize_weights' - (Bool):
    Whether or not to normalize the maximum edge weight to be 1.0, implictly
    alters the tolerance to which the problem is solved.

  Outputs
  -------
  * val - (Float):
    The value of the maximum weighted matching. 
  * noute - (Int):
    The cardinality of the matching found. 
  * match1 - (Array{Int,1}):
    The mapping linking left hand side nodes to the right hand side. The ith 
    entry maps node i to a right hand side node. 
  * match2 - (Array{Int,1}):
    The mapping linking right hand side nodes to the right hand side. The jth 
    entry maps node j to a left hand side node. The last n entries in the array
    represent dummy nodes which are used to account non-matches which may arise 
    from negative weights. 
------------------------------------------------------------------------------"""
function bipartite_matching_primal_dual(X::Union{Matrix{T},Adjoint{T,Matrix{T}}};primalDualTol::Float64=1e-8,
                                       normalize_weights::Bool=false) where T
    #to get the access pattern right, we must match the right hand side to the left hand side. 

	m,n = size(X)
	#@assert m >= n  #error occurs when m < n 
    if m < n 
        val, noute, B_to_A, _ = bipartite_matching_primal_dual(X';primalDualTol,normalize_weights)
        A_to_B = -ones(Int,m) # negative 1 to make no mistake that node is unmatched
        for (j,i) in enumerate(B_to_A)
            if i != -1
                A_to_B[i] = j
            end
        end
        return val, noute, A_to_B, B_to_A
    end


    # variables used for the primal-dual algorithm
    # normalize ai values # updated on 2-19-2019
    if normalize_weights
        X ./= maximum(abs.(X))
    end

	#initialize variables
    alpha=zeros(Float64,n)
    bt=zeros(Float64,m+n)#beta
    match1 = zeros(Int64,n)
    match2 = zeros(Int64,n+m)
    queue=zeros(Int64,m)
    t=zeros(Int64,m+n)
    tmod = zeros(Int64,m+n)
    ntmod=0

    # initialize the primal and dual variables

	for j = 1:n
		for i=1:m
			if X[i,j] > alpha[j]
			   alpha[j]=X[i,j]
			end
		end
    end


    # dual variables (bt) are initialized to 0 already
    # match1 and match2 are both 0, which indicates no matches

    j=1
    @inbounds while j<=n

        for i=1:ntmod
            t[tmod[i]]=0
        end
        ntmod=0
        # add i to the stack
        head=1
        tail=1
        queue[head]=j
        while head <= tail && match1[j]==0 #queue empty + i is unmatched
            
 		 	k=queue[head]
			#println("begining of queue loop")

            for i=1:m+1 #iterate over column k

				if i == m+1 #check the dummy node
					i = k+m
				end
				if i == k+m #dummy nodes don't have weight
					if 0.0 < alpha[k] + bt[i] - primalDualTol
						continue
					end
                elseif X[i,k] < alpha[k] + bt[i] - primalDualTol
					continue
                end # skip if tight

                if t[i]==0
                    tail=tail+1 #put the potential match in the queue
                    if tail <= m
                        queue[tail]=match2[i]
					end
                    t[i]=k  #try vertex k for vertex j
                    ntmod=ntmod+1
                    tmod[ntmod]=i
                    if match2[i]<1  #if i is unmatched
                        while i>0   #unfurl out to get an augmented path
                            match2[i]=t[i]
                            k=t[i]
                            temp=match1[k]
                            match1[k]=i
							i=temp
						end
                        break
                    end
                end
            end
            head=head+1

        end

        #if node j was unable to be matched, update flows and search for new augmenting path
		if match1[j] < 1
            theta=Inf
            for i=1:head-1
                t1=queue[i]
                for t2=1:m
                    if t[t2] == 0 && alpha[t1] + bt[t2] - X[t2,t1] < theta
                        theta = alpha[t1] + bt[t2] - X[t2,t1]
                    end
                end
				#check t1's dummy node
				if t[t1 + m] == 0 && alpha[t1] + bt[t1 + m] < theta
                        theta = alpha[t1] + bt[t1 + m]
				end
            end

            for i=1:head-1
                alpha[queue[i]] -= theta
            end
            for i=1:ntmod
                bt[tmod[i]] += theta
            end

            continue
        end

        j=j+1
    end

	#count
    val=0.0
    for j=1:n
        for i=1:m
            if i==match1[j]
                val=val+X[i,j]
            end
        end
    end

	#count how many are properly matched
    noute = 0
    for j=1:n
        if match1[j]<=m
            noute=noute+1
        end
	end
	
    A_to_B = -ones(Int,m)
    B_to_A = match1
    for (j,i) in enumerate(B_to_A)
        A_to_B[i] = j
    end

    return val, noute, A_to_B, B_to_A
end


#
# Local Search Methods
# 

function seq_similarity(U,V,matching)

	sim = 0.0
	for (i,ip) in matching 
		sim += dot(U[i,:],V[ip,:])
	end
	return sim
end

function TAME_score(A_motifs::Set{Vector{Int}},B::SymTensorUnweighted{Clique},
						B_edge_mask::Set{Int64},U::Matrix{T},V::Matrix{T}, 
						mapping::Dict{Int,Int}) where T

	#@assert maximum(keys(mapping)) <= B.n  # mapping oriented from B -> A

    #top_sim = 0
	motifs_matched = 0 


    for idx in B_edge_mask
		
        edge = B.indices[:,idx]
		
        new_edge = [get(mapping,i,-1) for i in edge]
        sort!(new_edge)

		#_,jp,kp = edge
		#_,j,k = new_edge

        if new_edge in A_motifs
		
			#top_sim += 1 + max(dot(U[j,:],V[jp,:]) + dot(U[k,:],V[kp,:]),
		    #					   dot(U[j,:],V[kp,:]) + dot(U[k,:],V[jp,:]))
			motifs_matched += 1 

        end
	

    end

    return top_sim, motifs_matched

end


function TAME_score(A_motifs::Set{Vector{Int}},B::SymTensorUnweighted{Clique},
					B_edge_mask::Set{Int64}, mapping::Dict{Int,Int}) where T


	motifs_matched = 0 

    for idx in B_edge_mask
		
        edge = B.indices[:,idx]
		
        new_edge = [get(mapping,i,-1) for i in edge]
        sort!(new_edge)

        if new_edge in A_motifs
		
			motifs_matched += 1 

        end

    end
    return motifs_matched

end

function TAME_score(A_motifs::Set{Vector{Int}},B::ThirdOrderSymTensor,
					B_edge_mask::Set{Int64}, mapping::Dict{Int,Int}) where T


	motifs_matched = 0 


    for idx in B_edge_mask
		
        edge = B.indices[:,idx]
		
        new_edge = [get(mapping,i,-1) for i in edge]
        sort!(new_edge)

        if new_edge in A_motifs
		
			motifs_matched += 1 

        end
	

    end


    return motifs_matched

end