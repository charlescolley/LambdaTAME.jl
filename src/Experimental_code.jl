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

function kron_contract_test(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,
                       U::Array{Float64,2},V::Array{Float64,2})
    n,d1 = size(U)
    m,d2 = size(V)
    @assert d1 == d2
    @assert A.n == n
    @assert B.n == m

    #result = zeros(A.cubical_dimension * B.cubical_dimension)

    ileave = (i,j) -> i + A.n*(j-1)

    max_rank = Int((d1-1)*d1/2+d1)
  #  result = zeros(n*m)
    println(Int(d1*(d1-1)/2))
	A_comps = zeros(n,Int(d1*(d1+1)/2))
	B_comps = zeros(m,Int(d1*(d1+1)/2))
	comp_idx = 1

    for i in 1:d1

        sub_A_i = tri_sub_tensor(A,U[:,i])
        sub_B_i = tri_sub_tensor(B,V[:,i])
        for j in 1:i

            A_update = (sub_A_i*U[:,j])
            B_update = (sub_B_i*V[:,j])

			A_comps[:,comp_idx] = A_update
			B_comps[:,comp_idx] = B_update
			comp_idx += 1
		    #=

			if i == j
				A_comps(:,comp_idx) = (sub_A_i*U[:,j])
				B_comps(:,comp_idx) = (sub_B_i*V[:,j])
               result += kron(B_update,A_update)
            else
                result += kron(2*B_update,A_update)
            end
            =#

        end

    end
	return A_comps, B_comps

  #  return reshape(result, A.n,B.n)
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


#  lowest_rank_TAME old code
function lowest_rank_TAME(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,W::Array{F,2},
                       β::F, max_iter::Int,tol::F,α::F;profile=false) where {F <:AbstractFloat}

    dimension = minimum((A.n,B.n))
    ranks = []

    if profile
        experiment_profile = Dict(
            "ranks"=>[],
            "contraction_timings"=>[],
            "svd_timings"=>[],
            "matching_timings"=>[],
            "hungarian_timings"=>[]
        )
    end

    best_triangle_count = -Inf
    best_x = copy(W)
    best_index = -1
    X_k = copy(W)
    X_k_1 = copy(W)
    i = 1
    lambda = Inf

    while true
        if profile
            (U,S,VT),t = @timed svd(X_k)
            singular_indexes = [i for i in 1:length(S) if S[i] > S[1]*eps(Float64)*dimension]
            push!(experiment_profile["ranks"],length(singular_indexes))
            push!(experiment_profile["svd_timings"],t)
        else
            (U,S,VT) = svd(X_k)
            singular_indexes = [i for i in 1:length(S) if S[i] > S[1]*eps(Float64)*dimension]
        end


        U = U[:,singular_indexes]
        V = VT[:,singular_indexes]*diagm(S[singular_indexes])

        if profile
            X_k_1,t = @timed kron_contract(A,B,U,V)
            push!(experiment_profile["contraction_timings"],t)
        else
            X_k_1 = kron_contract(A,B,U,V)
        end

        new_lambda = dot(X_k_1,X_k)

        if β != 0.0
            X_k_1 .+= β * X_k
        end

        if α != 1.0
            X_k_1 = α * X_k_1 + (1 - α) * W
        end

        X_k_1 ./= norm(X_k_1)

        if profile
            triangles, gaped_triangles, hungarian_time, matching_time = TAME_score(A,B,sparse(X_k_1);return_timings=true)
            push!(experiment_profile["hungarian_timings"],hungarian_time)
            push!(experiment_profile["matching_timings"],matching_time)
        else
            triangles, gaped_triangles =  TAME_score(A,B,X_k_1)
        end

        println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(gaped_triangles)")

        if triangles > best_triangle_count
            best_x = copy(X_k_1)
            best_triangle_count = triangles
            best_iterate = i
            best_U = copy(U)
            best_V = copy(V)
        end
        println("λ: $(new_lambda)")

        if abs(new_lambda - lambda) < tol || i >= max_iter
            if profile
                return best_x, best_triangle_count, ranks, experiment_profile
            else
                return best_x, best_triangle_count, ranks
            end
        else
            X_k = copy(X_k_1)
            lambda = new_lambda
            i += 1
        end

    end

    #compute the number of triangles matched in the last iterate

end