function kron_contract(A::COOTen,B::COOTen,
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

        sub_A_i = SparseSymmetricTensors.tri_sub_tensor(A,U[:,i])
        sub_B_i = SparseSymmetricTensors.tri_sub_tensor(B,V[:,i])
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

function implicit_contraction(A::COOTen,B::COOTen,x::Array{Float64,1})

    @assert length(x) == A.cubical_dimension*B.cubical_dimension
    m = A.cubical_dimension
    n = B.cubical_dimension
    y = zeros(Float64,length(x))

    ileave = (i,j) -> i + m*(j-1)

    for i in 1:length(A)
        i_1,i_2,i_3 = A.indices[i,:]
        for j in 1:length(B)
            j_1,j_2,j_3 = B.indices[j,:]

            y[ileave(i_1,j_1)] = 2*A.vals[i]*B.vals[j]*x[ileave(i_2,j_2)]*x[ileave(i_3,j_3)]
            y[ileave(i_2,j_2)] = 2*A.vals[i]*B.vals[j]*x[ileave(i_1,j_1)]*x[ileave(i_3,j_3)]
            y[ileave(i_3,j_3)] = 2*A.vals[i]*B.vals[j]*x[ileave(i_1,j_1)]*x[ileave(i_2,j_2)]
        end
    end

    return y
end

function low_rank_TAME(A::COOTen, B::COOTen,W::Array{F,2},rank::Int,
                       β::F, max_iter::Int,tol::F,α::F) where {F <:AbstractFloat}

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

      #  println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(gaped_triangles)")

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

#TODO: needs to be fixed
function align_tensors(A::COOTen,B::COOTen,rank::Int,method="ΛTAME")
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