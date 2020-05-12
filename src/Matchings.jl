

function low_rank_matching(U::Array{Float64,2},V::Array{Float64,2})
    n,d1 = size(U)
    m,d2 = size(V)
    @assert d1 == d2

    matching_weights = zeros(d1)
    matchings = Array{Dict{Int,Int},1}(undef,d1)

    for i = 1:d1
        matchings[i],matching_weights[i] = rank_one_matching(U[:,i],V[:,i])
    end

    D = (matching_weights.^(-1))*matching_weights'

    d_j = maximum(D,dims= 1)
    opt_j = argmin(d_j).I[2]

    return matchings[opt_j]

end


function rank_one_matching(u::Array{Float64,1},v::Array{Float64,1})

    #get maximum matching by rearrangement theorem
    u_perm = sortperm(u,rev=true)
    v_perm = sortperm(v,rev=true)

    matching_weight = 0
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


function search_Krylov_space(A::ssten.COOTen,B::ssten.COOTen,U::Array{Float64,2},V::Array{Float64,2})

    best_score = -1
    best_i = -1
    best_j = -1

    Triangle_check = Dict{Array{Int,1},Int}()

    if A.unique_nnz > B.unique_nnz
        for i in 1:A.unique_nnz
            Triangle_check[A.indices[i,:]] = 1
        end
        Input_tensor = B
    else
        for i in 1:B.unique_nnz
            Triangle_check[B.indices[i,:]] = 1
        end
        Input_tensor = A
    end

    for i in 1:size(U,2)
       for j in 1:size(V,2)

            if A.unique_nnz > B.unique_nnz
                matched_tris,gaped_tris = TAME_score(Triangle_check,Input_tensor,V[:,j],U[:,i])
            else
                matched_tris,gaped_tris = TAME_score(Triangle_check,Input_tensor,U[:,i],V[:,i])
            end

            if matched_tris > best_score
                best_score = matched_tris
                best_i = i
                best_j = j
            end
        end
    end
    return best_score, best_i, best_j
end

#used when we don't want to recreate the triangle matching dictionary multiple times
function TAME_score(Triangle_Dict::Dict{Array{Int,1},Int},Input_tensor::ssten.COOTen,
                    u::Array{Float64,1},v::Array{Float64,1}) where {Ten <: ssten.AbstractSSTen}

    Match_mapping, _ = rank_one_matching(u,v)
    TAME_score(Triangle_Dict,Input_tensor,Match_mapping)

end

#Computes the TAME score for this iterate by
function TAME_score(A::ssten.COOTen,B::ssten.COOTen,u::Array{Float64,1},v::Array{Float64,1}) where {Ten <: ssten.AbstractSSTen}

    Match_mapping, _ = rank_one_matching(u,v)
    TAME_score(A,B,Match_mapping)

end

function TAME_score(A::ssten.COOTen,B::ssten.COOTen,
                    U::Array{Float64,2},V::Array{Float64,2}) where {Ten <: ssten.AbstractSSTen}
    println(size(U),"  ",size(V))
    Match_mapping = low_rank_matching(U,V)
  #  println(Match_mapping)
    TAME_score(A,B,Match_mapping)

end

function TAME_score(A::ssten.COOTen,B::ssten.COOTen,Match_mapping::Dict{Int,Int}) where {Ten <: ssten.AbstractSSTen}

    match_len = length(Match_mapping)

    Triangle_check = Dict{Array{Int,1},Int}()
    gaped_triangles = 0
    triangle_count = 0

    if A.unique_nnz > B.unique_nnz

        for i in 1:A.unique_nnz
            Triangle_check[A.indices[i,:]] = 1
        end

        #invert to map v indices to u
        Match_mapping = Dict(value => key for (key, value) in Match_mapping)

        for i in 1:B.unique_nnz
            v_i,v_j,v_k = B.indices[i,:]

            matched_triangle =
              sort([get(Match_mapping,v_i,-1),get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])
    #        println(B.indices[i,:]," -> ",matched_triangle)
            match = get(Triangle_check,matched_triangle,0)
            if match == 1
                triangle_count += 1
            else
                gaped_triangles += 1
            end
        end

    else
        for i in 1:B.unique_nnz
            Triangle_check[B.indices[i,:]] = 1
        end

        for i in 1:A.unique_nnz
            v_i,v_j,v_k = A.indices[i,:]
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

    return triangle_count, gaped_triangles #sub_A, sub_B

end

function TAME_score(Triangle_Dict::Dict{Array{Int,1},Int},Input_tensor::ssten.COOTen,
                    Match_mapping::Dict{Int,Int}) where {Ten <: ssten.AbstractSSTen}

    triangle_count = 0
    gaped_triangles = 0

    for i in 1:Input_tensor.unique_nnz
        v_i,v_j,v_k = Input_tensor.indices[i,:]

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

