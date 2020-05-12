



function align_tensors(A::ssten.COOTen,B::ssten.COOTen)
    iter =15
    tol=1e-6
    rank = 10
    alphas = [.15,.5,.85]
    betas =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001]

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))

    TAME_timings = Array{Float64,1}(undef,length(alphas)*length(betas))
    Krylov_Search_timings = Array{Float64,1}(undef,length(alphas)*length(betas))

    U = Array{Float64,2}(undef,A.cubical_dimension,iter)
    V = Array{Float64,2}(undef,B.cubical_dimension,iter)
    best_TAME_PP_tris = -1
    best_i  = -1
    best_j = -1
    exp_index = 1

    for α in alphas
        for beta in betas
 #           ((TAME_u, TAME_v, U, V),runtime) = @timed TAME(A,B,beta,iter,tol,α)


            ((_, best_U, best_V, best_triangle_count),runtime) = @timed low_rank_TAME(A, B,ones(A.cubical_dimension,B.cubical_dimension),rank,beta, 15,tol,α)
            ((U,V),runtime) = @timed TAME(A,B,beta,iter,tol,α)
            TAME_timings[exp_index] = runtime
            search_tris, _  = TAME_score(A,B,best_U,Matrix(best_V'))
            #TAME_tris, _= TAME_score(A,B,TAME_u,TAME_v)

            #search the Krylov Subspace
 #           ((search_tris, i, j),runtime) = @timed search_Krylov_space(A,B,U,V)
 #           Krylov_Search_timings[exp_index] = runtime
#            println("α: $α -- β:$beta -- OG TAME: $TAME_tris -- TAME++: $search_tris")

            if search_tris > best_TAME_PP_tris
                best_TAME_PP_tris = search_tris
                best_i = i
                best_j = j
            end
            exp_index += 1
        end

    end

    avg_TAME_timings = sum(TAME_timings)/length(TAME_timings)
    avg_Krylov_timings = sum(Krylov_Search_timings)/length(Krylov_Search_timings)

    return best_TAME_PP_tris,max_triangle_match, avg_TAME_timings, avg_Krylov_timings
end

function align_tensors(A::ssten.COOTen,B::ssten.COOTen,rank::Int)
    iter =15
    tol=1e-6
    alphas = [.15,.5,.85]
    betas =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001]

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))

    TAME_timings = Array{Float64,1}(undef,length(alphas)*length(betas))
    Krylov_Search_timings = Array{Float64,1}(undef,length(alphas)*length(betas))

    U = Array{Float64,2}(undef,A.cubical_dimension,iter)
    V = Array{Float64,2}(undef,B.cubical_dimension,iter)

    best_TAME_PP_tris = -1
    best_i  = -1
    best_j = -1
    exp_index = 1

    for α in alphas
        for beta in betas
            ((_, best_U, best_V, best_triangle_count),runtime) = @timed low_rank_TAME(A, B, ones(A.cubical_dimension,B.cubical_dimension),rank,beta, 15,tol,α)
            TAME_timings[exp_index] = runtime
            search_tris, _  = TAME_score(A,B,best_U,Matrix(best_V'))

            if search_tris > best_TAME_PP_tris
                best_TAME_PP_tris = search_tris
                best_i = i
                best_j = j
            end
            exp_index += 1
        end

    end

    avg_TAME_timings = sum(TAME_timings)/length(TAME_timings)
    avg_Krylov_timings = sum(Krylov_Search_timings)/length(Krylov_Search_timings)

    return best_TAME_PP_tris,max_triangle_match, avg_TAME_timings, avg_Krylov_timings
end

function TAME(A::ssten.COOTen, B::ssten.COOTen, β::Float64, max_iter::Int,tol::Float64,α::Float64) where {Ten <: ssten.AbstractSSTen}

    U = zeros(A.cubical_dimension,max_iter)
    V = zeros(B.cubical_dimension,max_iter)

   # u = initialize_initial(A)
   # v = initialize_initial(B)
    u_0 = ones(A.cubical_dimension)
    u_0 /=norm(u_0)

    v_0 = ones(B.cubical_dimension)
    v_0 /=norm(v_0)


    u = copy(u_0)
    v = copy(v_0)

    sqrt_β = β^(.5)

    u /= norm(u)
    v /= norm(v)

    lambda = Inf

    #initialize space for the best solutions
    best_u = zeros(Float64,A.cubical_dimension)
    best_v = zeros(Float64,B.cubical_dimension)
    best_triangle_count = 0
    best_iterate = -1

    i = 1

    while true
        u_new = ssten.contract_k_1(A,u)
        v_new = ssten.contract_k_1(B,v)

        lambda_A = (u_new'*u)
        lambda_B = (v_new'*v)
        new_lambda = lambda_A*lambda_B

        if β != 0.0
            u_new .+= sqrt_β*u
            v_new .+= sqrt_β*v
        end

        if α != 1.0
            u_new = α*u_new + (1 -α)*u_0
            v_new = α*v_new + (1 -α)*v_0
        end

        u_new ./= norm(u_new)
        v_new ./= norm(v_new)
#        triangles, gaped_triangles = TAME_score(A,B,u,v)

   #     println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(gaped_triangles)")

#        if triangles > best_triangle_count
#            best_u = copy(u)
#            best_v = copy(v)
#            best_triangle_count = triangles
#            best_iterate = i
#        end

   #     println("λ_A: $(lambda_A) -- λ_B: $(lambda_B) -- newλ: $(new_lambda)")

        U[:,i] = u
        V[:,i] = v
        if abs(new_lambda - lambda) < tol || i >= max_iter
            return U, V
        else

            lambda = new_lambda
            u = copy(u_new)
            v = copy(v_new)
            i += 1
        end

    end

end


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
                matched_tris,gaped_tris = TAME_score(Triangle_check,Input_tensor,U[:,i],V[:,j])
            else
                matched_tris,gaped_tris = TAME_score(Triangle_check,Input_tensor,V[:,j],U[:,i])
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

    #invert to map v indices to u
    Match_mapping = Dict(value => key for (key, value) in Match_mapping)

    for i in 1:Input_tensor.unique_nnz
        v_i,v_j,v_k = Input_tensor.indices[i,:]

        matched_triangle =
          sort([get(Match_mapping,v_i,-1),get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])

        match = get(Triangle_check,matched_triangle,0)
        if match == 1
            triangle_count += 1
        else
            gaped_triangles += 1
        end
    end
    return triangle_count, gaped_triangles
end


function produce_ssten_from_triangles(file)

    A = MatrixNetworks.readSMAT(file)
    (n,m) = size(A)
    if(n != m)
        println("recatangular")
    end

    T = collect(MatrixNetworks.triangles(A))

    output_file = alterfilename(file,".ssten",".",false)

    open(output_file,"w") do f
        write(f,"$(3)\t$(n)\t$(length(T))\n")

        for (v_i,v_j,v_k) in T
            write(f,"$(v_i)\t$(v_j)\t$(v_k)\t1.0\n")
        end
    end

end

function low_rank_TAME(A::ssten.COOTen, B::ssten.COOTen,W::Array{F,2},rank::Int,
                       β::F, max_iter::Int,tol::F,α::F) where {Ten <: ssten.AbstractSSTen,F <:AbstractFloat}

    #low rank factorization
    #U,S,VT = svd(W)
    #U = U[:,1:rank]
    #V = diagm(S)*VT'[:,1:rank]
    #TODO: fix this to compute top k vectors
    U,V = NMF.randinit(W,rank)
    V = Matrix(V')

    best_triangle_count,_ = TAME_score(A,B,U,V)
    best_U = copy(U)
    best_V = copy(V)
    best_index = 1


    #w = reshape(W, A.cubical_dimension,B.cubical_dimension)
    #get the low rank factors of W

    x_k = copy(W)
    i = 1
    lambda = Inf
    while true

        x_k_1 = kron_contract(A,B,U,V)
        new_lambda = dot(x_k_1,x_k)

        if β != 0.0
            println(size(x_k))
            println(size(x_k_1))
            x_k_1 .+= β * x_k
        end

        if α != 1.0
            x_k_1 = α * x_k_1 + (1 - α) * W
        end

        x_k_1 ./= norm(x_k_1)

        #low rank factorization
        #TODO: fix this to compute top k vectors

        U,V = NMF.randinit(x_k_1,rank)
        V = Matrix(V')
#        U,S,VT = svd(reshape(x_k_1,A.cubical_dimension,B.cubical_dimension))
#        U = U[:,1:rank]
#        V = diagm(S)*VT'[:,1:rank]

        triangles, gaped_triangles = TAME_score(A,B,U,V)

        println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(gaped_triangles)")

        if triangles > best_triangle_count
            best_x = copy(x_k_1)
            best_triangle_count = triangles
            best_iterate = i
            best_U = copy(U)
            best_V = copy(V)
        end

        println("λ: $(new_lambda)")

        if abs(new_lambda - lambda) < tol || i >= max_iter
            return x_k_1, best_U, best_V, best_triangle_count
        else

            lambda = new_lambda
            i += 1
        end

    end

end

function kron_contract(A::ssten.COOTen,B::ssten.COOTen,
                       U::Array{Float64,2},V::Array{Float64,2})
    n,d1 = size(U)
    m,d2 = size(V)
#    @assert d1 == d2
#    @assert A.cubical_dimension == n
#    @assert B.cubical_dimension == m

#    result = zeros(A.cubical_dimension * B.cubical_dimension)

    max_rank = Int((d1-1)*d1/2+d1)
    result = zeros(n,m)

    for i in 1:d1

        sub_A_i = ssten.tri_sub_tensor(A,U[:,i])
        sub_B_i = ssten.tri_sub_tensor(B,V[:,i])
        for j in 1:i

            A_update = (sub_A_i*U[:,j])
            B_update = (sub_B_i*V[:,j])


            if i == j
                for i_1=1:n
                    for i_2=1:m
                        result[i_1,i_2] += A_update[i_1]*B_update[i_2]
                     end
                 end
            else
                for i_1=1:n
                    for i_2=1:m
                        result[i_1,i_2] += 2*A_update[i_1]*B_update[i_2]
                    end
                end
            end

        end
    end

    return result
end
