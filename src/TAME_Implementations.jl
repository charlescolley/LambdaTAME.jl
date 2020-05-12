

function align_tensors(A::ssten.COOTen,B::ssten.COOTen)
    iter =15
    tol=1e-6
    rank = 10
    alphas = [.15,.5,.85]
    betas =[1000.0,100.0,10.0,1.0,0.0,0.1,0.01,0.001]

    max_triangle_match = min(size(A.indices,1),size(B.indices,1))
    total_triangles = size(A.indices,1) + size(B.indices,1)

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
            ((U,V),runtime) = @timed ΛTAME(A,B,beta,iter,tol,α)
            TAME_timings[exp_index] = runtime

            #search the Krylov Subspace
            ((search_tris, i, j),runtime) = @timed search_Krylov_space(A,B,U,V)
            Krylov_Search_timings[exp_index] = runtime

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

    return best_TAME_PP_tris,max_triangle_match,total_triangles, avg_TAME_timings, avg_Krylov_timings
end

#TODO: needs to be fixed
function align_tensors(A::ssten.COOTen,B::ssten.COOTen,rank::Int,method="ΛTAME")
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

            if method == "ΛTAME"
                U,V = TAME(A,B,beta,iter,tol,α)

            else
                ((_, best_U, best_V, best_triangle_count),runtime) = @timed low_rank_TAME(A, B, ones(A.cubical_dimension,B.cubical_dimension),rank,beta, iter,tol,α)
            end
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

function ΛTAME(A::ssten.COOTen, B::ssten.COOTen, β::Float64, max_iter::Int,tol::Float64,α::Float64) where {Ten <: ssten.AbstractSSTen}

    U = zeros(A.cubical_dimension,max_iter+1)
    V = zeros(B.cubical_dimension,max_iter+1) #store initial in first column

    U[:,1] = ones(A.cubical_dimension)
    U[:,1] /=norm(U[:,1])

    V[:,1] = ones(B.cubical_dimension)
    V[:,1] /=norm(U[:,1])

    sqrt_β = β^(.5)

    lambda = Inf
    i = 1

    while true

        U[:,i+1] = ssten.contract_k_1(A,U[:,i])
        V[:,i+1] = ssten.contract_k_1(B,V[:,i])

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

        println("iteration $(i)    λ_A: $(lambda_A) -- λ_B: $(lambda_B) -- newλ: $(new_lambda)")

        if abs(new_lambda - lambda) < tol || i >= max_iter
            return U[:,1:i], V[:,1:i]
        else
            lambda = new_lambda
            i += 1
        end

    end

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
