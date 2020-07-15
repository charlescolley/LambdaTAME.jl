
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

function search_Krylov_space(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,U::Array{Float64,2},V::Array{Float64,2})

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

function search_Krylov_space(A::COOTen,B::COOTen,U::Array{Float64,2},V::Array{Float64,2})

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
function TAME_score(Triangle_Dict::Dict{Array{Int,1},Int},Input_tensor::COOTen,
                    u::Array{Float64,1},v::Array{Float64,1})

    Match_mapping, _ = rank_one_matching(u,v)
    TAME_score(Triangle_Dict,Input_tensor,Match_mapping)

end


#Computes the TAME score for this iterate by
function TAME_score(A::COOTen,B::COOTen,u::Array{Float64,1},v::Array{Float64,1})

    Match_mapping, _ = rank_one_matching(u,v)
    TAME_score(A,B,Match_mapping)

end

function TAME_score(A::COOTen,B::COOTen,U::Array{Float64,2},V::Array{Float64,2})

    Match_mapping = low_rank_matching(U,V)
    TAME_score(A,B,Match_mapping)

end

function TAME_score(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,U::Array{Float64,2},V::Array{Float64,2})

    Match_mapping = low_rank_matching(U,V)
    TAME_score(A,B,Match_mapping)

end

function TAME_score(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor, X::SparseMatrixCSC{Float64,Int64};return_timings=false)

    if return_timings
        x ,bipartite_matching_time = @timed bipartite_matching(X) #negate because hungarian finds minimum weight matching
        (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,Dict(i => j for (i,j) in enumerate(x.match)))
        return triangle_count, gaped_triangles, bipartite_matching_time, scoring_time
    else
        x = bipartite_matching(X) #negate because hungarian finds minimum weight matching
        return TAME_score(A,B,Dict(i => j for (i,j) in enumerate(x.match)))
    end

end

#run the Hungarian algorithm on an iterate
function TAME_score(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,X::Array{Float64,2};return_timings=false)

    if return_timings
        (matching, _) ,hungarian_time = @timed hungarian(-X) #negate because hungarian finds minimum weight matching
        (triangle_count, gaped_triangles), matching_time = @timed TAME_score(A,B,Dict(i => j for (i,j) in enumerate(matching)))
        return triangle_count, gaped_triangles, hungarian_time, matching_time
    else
        matching, _  = hungarian(-X) #negate because hungarian finds minimum weight matching
        return TAME_score(A,B,Dict(i => j for (i,j) in enumerate(matching)))
    end
end

function TAME_score(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,x::Array{Float64,1};return_timings=false)


    if return_timings
        (matching, _) ,hungarian_time = @timed hungarian(reshape(-x,A.n,B.n)) #negate because hungarian finds minimum weight matching
        (triangle_count, gaped_triangles), matching_time = @timed TAME_score(A,B,Dict(i => j for (i,j) in enumerate(matching)))
        return triangle_count, gaped_triangles, hungarian_time, matching_time
    else
        matching, _  = hungarian(reshape(-x,A.n,B.n)) #negate because hungarian finds minimum weight matching
        return TAME_score(A,B,Dict(i => j for (i,j) in enumerate(matching)))
    end

end

function TAME_score(A::COOTen,B::COOTen,X::Array{Float64,2};return_timings=false)


   if return_timings
        (matching, _) ,hungarian_time = @timed hungarian(-X) #negate because hungarian finds minimum weight matching
        (triangle_count, gaped_triangles), matching_time = @timed TAME_score(A,B,Dict(i => j for (i,j) in enumerate(matching)))
        return triangle_count, gaped_triangles, hungarian_time, matching_time
    else
        matching, _  = hungarian(-X) #negate because hungarian finds minimum weight matching
        return TAME_score(A,B,Dict(i => j for (i,j) in enumerate(matching)))
    end

end

function TAME_score(A::COOTen,B::COOTen,x::Array{Float64,1};return_timings=false)

   if return_timings
        (matching, _) ,hungarian_time = @timed hungarian(-reshape(-x,A.cubical_dimension,B.cubical_dimension)) #negate because hungarian finds minimum weight matching
        (triangle_count, gaped_triangles), matching_time = @timed TAME_score(A,B,Dict(i => j for (i,j) in enumerate(matching)))
        return triangle_count, gaped_triangles, hungarian_time, matching_time
    else
        matching, _  = hungarian(reshape(-x,A.cubical_dimension,B.cubical_dimension)) #negate because hungarian finds minimum weight matching
        return TAME_score(A,B,Dict(i => j for (i,j) in enumerate(matching)))
    end

end

function TAME_score(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,Match_mapping::Dict{Int,Int})

    match_len = length(Match_mapping)

    Triangle_check = Dict{Array{Int,1},Int}()
    gaped_triangles = 0
    triangle_count = 0

    if size(A.indices,1) > size(B.indices,1)

        for i in 1:size(A.indices,1)
            Triangle_check[A.indices[i,:]] = 1
        end

        #invert to map v indices to u
        Match_mapping = Dict(value => key for (key, value) in Match_mapping)

        for i in 1:size(B.indices,1)
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
        for i in 1:size(B.indices,1)
            Triangle_check[B.indices[i,:]] = 1
        end

        for i in 1:size(A.indices,1)
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

function TAME_score(A::COOTen,B::COOTen,Match_mapping::Dict{Int,Int})

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

function TAME_score(Triangle_Dict::Dict{Array{Int,1},Int},Input_tensor::COOTen,
                    Match_mapping::Dict{Int,Int})

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

