
#fix iterations
function lowest_rank_TAME_test(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
                          U_0::Array{F,2},V_0::Array{F,2},
                          β::F, max_iter::Int,tol::F,α::F;profile=false) where {F <:AbstractFloat}

	@assert size(U_0,2) == size(V_0,2)

    dimension = minimum((A.n,B.n))

    best_triangle_count = -Inf
    best_x = zeros(size(U_0,1),size(V_0,1))
    best_index = -1

	r = 1
	X_k = U_0 * V_0'
	normalization_factor = norm(X_k)
	#TODO: fix factorization
	X_k ./= normalization_factor

	X_0 = copy(X_k)
	U_0 ./= normalization_factor
#	V_0 ./= sqrt(normalization_factor)

	U_k = deepcopy(U_0)
	V_k = deepcopy(V_0)


	All_vectors = zeros(A.n,B.n,max_iter)

	i = 1
    lambda = Inf

    if profile
        experiment_profile = Dict(
            "ranks"=>[],
            "contraction_timings"=>[],
            "svd_timings"=>[],
            "matching_timings"=>[],
            "scoring_timings"=>[]
        )
    end

    for _ in 1:max_iter

#		println("U_k",U_k[1:2,:])
		println("size of U_0",size(U_0))
		println("size of U_k:",size(U_k))

		#println("|U_k - U_0|_F = ",norm(U_k - U_0))
        if profile
            #X_k_1,t = @timed kron_contract(A,B,U_k,V_k)
		    (A_comps, B_comps),t = @timed get_kron_contract_comps(A,B,U_k,V_k)
            push!(experiment_profile["contraction_timings"],t)
        else
            A_comps, B_comps = get_kron_contract_comps(A,B,U_k,V_k)
			# = size(U_k,2)^2 + size(U_k,2) + size(U_0,2)
			X_k_1 = kron_contract(A,B,sparse(X_k),r)
			println("|X_{k_1} - T_A*T_B'|_F=",norm(X_k_1 - A_comps*B_comps')/norm(X_k_1))
        end

		if α != 1.0 && β != 0.0
			U_temp = hcat(sqrt(α) * A_comps, sqrt(α) * sqrt(β) * U_k, sqrt(1-α) * U_0)
			V_temp = hcat(sqrt(α) * B_comps, sqrt(α) * sqrt(β) * V_k, sqrt(1-α) * V_0)
#			U_temp = hcat(α * A_comps, α * β * U_k, (1-α) * U_0)
#			V_temp = hcat(B_comps, V_k, V_0)
		elseif α != 1.0
			U_temp = hcat(sqrt(α)*A_comps, sqrt(1-α)*U_0)
			V_temp = hcat(sqrt(α)*B_comps, sqrt(1-α)*V_0)
		elseif β != 1.0
			U_temp = hcat(A_comps, sqrt(β) * U_k)
			V_temp = hcat(B_comps, sqrt(β) * V_k)
		else
			U_temp = A_comps
			V_temp = B_comps
		end

		println("U_temp size:",size(U_temp))
		println("U_temp rank:",rank(U_temp))
		X_k_1 = α*(X_k_1 + β*X_k) + (1-α)*X_0
	#	if i > 1
	#		return X_k_1 - U_temp*V_temp'
	#	end


		println("|(α*(X_k_1 + β*X_k) + (1-α)*X_0) - U_temp*V_temp'|_F=",norm(X_k_1 - U_temp*V_temp')/norm(X_k_1))

       # Q_A,R_A = qr(U_temp)
       # Q_B,R_B = qr(V_temp)

		#@show svdvals(U_temp*V_temp')
		#@show  svdvals(R_A*R_B')

    #    _,test_singvals,_ = svd(R_A*R_B')
     #   _,normalized_singvals,_ = svd((R_A*R_B')/norm(R_A*R_B'))

        #println("singular values of R_A':",test_singvals)
    #    println("singular values of R_A'R_B:",normalized_singvals)
#		println("trace of R_A'R_B:",sum(normalized_singvals))
       #println("singular values of R_B':",RS_B)

		 if profile
            #X_k_1,t = @timed kron_contract(A,B,U_k,V_k)
			(result_A),t_A = @timed svd(U_temp)
			#println(typeof(result_A))
			A_U,A_S,A_Vt = result_A.U,result_A.S,result_A.Vt
			(result_B),t_B = @timed svd(V_temp)
			B_U,B_S,B_Vt = result_B.U,result_B.S,result_B.Vt
            push!(experiment_profile["svd_timings"],t_A + t_B)
        else
			A_U,A_S,A_Vt = svd(U_temp)
			B_U,B_S,B_Vt = svd(V_temp)
		end

		#println("OG A singular values:",A_S)
		#println("OG B singular values:",B_S)
		singular_indexes_A = [i for i in 1:length(A_S) if A_S[i] > A_S[1]*eps(Float64)*size(U_temp,2)]
		singular_indexes_B = [i for i in 1:length(B_S) if B_S[i] > B_S[1]*eps(Float64)*size(V_temp,2)]

	#	println("length of singular_indexes_A:",length(singular_indexes_A))
		U_k_1 = A_U[:,singular_indexes_A]
#		U_k_1 = A_U*(diagm(A_S)*A_Vt')

		V_k_1 = V_temp*(A_Vt[:,singular_indexes_A]*diagm(A_S[singular_indexes_A]))


#		println("U_k_1 cond:",cond(U_k_1))
#		println("V_k_1 cond:",cond(V_k_1))
	#	V_k_1 ./= norm(V_k_1)



		#=
		core = diagm(A_S[singular_indexes_A])*(A_Vt[:,singular_indexes_A]'*B_Vt[:,singular_indexes_B])*diagm(B_S[singular_indexes_B])
		#println("core size is:",size(core))
		core ./= norm(core)  #normalizing the core is cheaper than the other components
		U_k_1 = A_U[:,singular_indexes_A]
		#V_k_1 = V_temp*(A_Vt[:,singular_indexes_A]*diagm(A_S[singular_indexes_A]))
		V_k_1 = B_U[:,singular_indexes_B]*core'
		=#
		#new_lambda = dot(V_k_1'*V_k, U_k_1'*U_k)


		#U_k_1 = A_U[:,singular_indexes_A]*(diagm(A_S[singular_indexes_A])*A_Vt[:,singular_indexes_A]')
		#V_k_1 = B_U[:,singular_indexes_B]*(diagm(B_S[singular_indexes_B])*B_Vt[:,singular_indexes_B]')


		#println(norm(X_k_1))




#		normalization_factor = norm(X_k_1)
		#X_k_1 = U_k_1*V_k_1'

		C = norm(X_k_1)
		println("|X_k_1 - U_k_1*V_k_1'|_F=",norm(X_k_1 - U_k_1*V_k_1')/C)
		X_k_1 ./= C

		C = norm(U_k_1*V_k_1')
		U_k_1 ./= sqrt(C)
		V_k_1 ./= sqrt(C)

		All_vectors[:,:,i] = U_k_1*V_k_1'

		r = rank(X_k_1)
		println("rank of X_k_1:       ",r)
		println("rank of U_k_1*V_k_1':",rank(All_vectors[:,:,i]))
#		V_k_1 ./= sqrt(normalization_factor)
#		X_k_1 ./= normalization_factor

		lam = dot(X_k_1,kron_contract(A,B,U_k_1,V_k_1))
		#println("old lambda:$new_lambda")
#


	#   NOTE:   triangle_matching is turned off for speed
# 		println(norm(X_k_1))
	#	sparse_X_k_1 = sparse(X_k_1)

        if profile
            triangles, gaped_triangles, matching_time, scoring_time = TAME_score(A,B,sparse_X_k_1;return_timings=true)
            push!(experiment_profile["matching_timings"],matching_time)
            push!(experiment_profile["scoring_timings"],scoring_time)
        else
     #       triangles, gaped_triangles = TAME_score(A,B,sparse_X_k_1)
        end

      #=  println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(gaped_triangles)")

        if triangles > best_triangle_count
            best_x = deepcopy(X_k_1)
            best_triangle_count = triangles
            best_iterate = i
        end
		=#
        println("λ_i: $(lam)")

        if abs(lam - lambda) < tol || i >= max_iter
            if profile
                return best_x, best_triangle_count, experiment_profile
            else
                return best_x, best_triangle_count, All_vectors
            end
        end

		#get the low rank factorization for the next one
		U_k = copy(U_k_1)
		V_k = copy(V_k_1)
		X_k = copy(X_k_1)

		lambda = lam
		i += 1

		println("U_k:",U_k)
		println("norm of U_k*V_k' at end:",norm(U_k*V_k'))
		#println("norm of V_k at end:",norm(V_k))
		println("norm of X_k at end:",norm(X_k))

    end

end

function low_rank_TAME_contraction(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
                                   U_k::Array{F,2},V_k::Array{F,2},α::Float64,β::Float64,
                                   U_0::Array{F,2},V_0::Array{F,2};profile=false) where {F <:AbstractFloat}

	if profile
		#X_k_1,t = @timed kron_contract(A,B,U_k,V_k)
		(A_comps, B_comps),t = @timed get_kron_contract_comps(A,B,U_k,V_k)
#		push!(experiment_profile["contraction_timings"],t)
	else
		A_comps, B_comps = get_kron_contract_comps(A,B,U_k,V_k)
	end

	#add in shift and mixing terms
	if α != 1.0 && β != 0.0
		U_temp = hcat(sqrt(α)*A_comps,sqrt(α*β)*U_k, sqrt(1-α)*U_0)
		#V_temp = hcat(B_comps,V_k,V_0)
		V_temp = hcat(sqrt(α)*B_comps,sqrt(α*β)*V_k, sqrt(1-α)*V_0)
	elseif α != 1.0
		U_temp = hcat(sqrt(α)*A_comps, sqrt(1-α)*U_0)
		V_temp = hcat(sqrt(α)*B_comps, sqrt(1-α)*V_0)
	elseif β != 1.0
		U_temp = hcat(A_comps, sqrt(β)*U_k)
		V_temp = hcat(B_comps, sqrt(β)*V_k)
	else
		U_temp = A_comps
		V_temp = B_comps
	end

	return U_temp , V_temp
	if profile
		#X_k_1,t = @timed kron_contract(A,B,U_k,V_k)
		(result_A),t_A = @timed svd(U_temp)
		A_U,A_S,A_Vt = result_A.U,result_A.S,result_A.Vt
		(result_B),t_B = @timed svd(V_temp)
		B_U,B_S,B_Vt = result_B.U,result_B.S,result_B.Vt
	#	push!(experiment_profile["svd_timings"],t_A + t_B)
	else
		A_U,A_S,A_Vt = svd(U_temp)
		#B_U,B_S,B_Vt = svd(V_temp)
	end
	#println(A_S)
	#println(B_S)
	singular_indexes_A = [i for i in 1:length(A_S) if A_S[i] > A_S[1]*eps(Float64)*A.n]
	#singular_indexes_B = [i for i in 1:length(B_S) if B_S[i] > B_S[1]*eps(Float64)*B.n]
	println(length(singular_indexes_A))
	#println(length(singular_indexes_B))


	#core = diagm(A_S[singular_indexes_A])*(A_Vt[:,singular_indexes_A]'*B_Vt[:,singular_indexes_B])*diagm(B_S[singular_indexes_B])
	#core ./= norm(core) #normalize
	U_k_1 = A_U[:,singular_indexes_A]
	V_k_1 = V_temp*(A_Vt[:,singular_indexes_A]*diagm(A_S[singular_indexes_A]))

	return U_k_1, V_k_1
end


#TODO: would be nice to load in processes as needed
#TODO: need a more robust way to load in the functions,
#          currently only works while running w/in directory
#TODO: check type stability
function lowest_rank_TAME_parallel_test(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
                          U_0::Array{F,2},V_0::Array{F,2},
                          β::F, max_iter::Int,tol::F,α::F;profile=false) where {F <:AbstractFloat}

	@assert size(U_0,2) == size(V_0,2)


	matching_proc = addprocs(1)
	futures = [] #TODO: can we type this?
	#load the functions on process
	@everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")


    dimension = minimum((A.n,B.n))

	X_k = U_0 * V_0'
	X_0 = copy(X_k)

	U_k = copy(U_0)
	V_k = copy(V_0)

	U_k ./= norm(U_k)
	V_k ./= norm(V_k)

	i = 1
    lambda = Inf

    if profile
        experiment_profile = Dict(
            "ranks"=>[],
            "contraction_timings"=>[],
            "svd_timings"=>[],
            "matching_timings"=>[],
            "scoring_timings"=>[]
        )
    end

    while true

        if profile
            #X_k_1,t = @timed kron_contract(A,B,U_k,V_k)
		    (A_comps, B_comps),t = @timed get_kron_contract_comps(A,B,U_k,V_k)
            push!(experiment_profile["contraction_timings"],t)
        else
            A_comps, B_comps = kron_contract_test2(A,B,U_k,V_k)
        end

		U_temp = hcat(sqrt(α)*A_comps,sqrt(α*β)*U_k, sqrt(1-α)*U_0)
		V_temp = hcat(sqrt(α)*B_comps,sqrt(α*β)*V_k, sqrt(1-α)*V_0)

		 if profile
			#X_k_1,t = @timed kron_contract(A,B,U_k,V_k)
			(result_A),t_A = @timed svd(U_temp)
			#println(typeof(result_A))
			A_U,A_S,A_Vt = result_A.U,result_A.S,result_A.Vt
			(result_B),t_B = @timed svd(V_temp)
			B_U,B_S,B_Vt = result_B.U,result_B.S,result_B.Vt
			push!(experiment_profile["svd_timings"],t_A + t_B)
        else
			A_U,A_S,A_Vt = svd(U_temp)
			B_U,B_S,B_Vt = svd(V_temp)
		end
		singular_indexes_A = [i for i in 1:length(A_S) if A_S[i] > A_S[1]*eps(Float64)*dimension]
		singular_indexes_B = [i for i in 1:length(B_S) if B_S[i] > B_S[1]*eps(Float64)*dimension]

		U_k_1 = A_U[:,singular_indexes_A]*diagm(A_S[singular_indexes_A])
		V_k_1 = (B_U[:,singular_indexes_B]*(diagm(B_S[singular_indexes_B])*B_Vt[:,singular_indexes_B]'*A_Vt[:,singular_indexes_A]))

		U_k_1 ./= norm(U_k_1)
		V_k_1 ./= norm(V_k_1)

		#run Triangle matching on another process
		f = @spawnat matching_proc[1] produce_scored_matching(A,B,U_k_1,U_k_1;return_timings=true)
		#TODO: check memory usage
		push!(futures,f)

		new_lambda = dot(V_k_1'*V_k, U_k_1'*U_k)

        println("finished iterate $(i):λ_i = $new_lambda")

        println("λ: $(new_lambda)")

        if abs(new_lambda - lambda) < tol || i >= max_iter
			break
        else
			#get the low rank factorization for the next one
			U_k = copy(U_k_1)
			V_k = copy(V_k_1)

			X_k = copy(X_k_1)
			lambda = new_lambda
			i += 1
        end

    end

	#check and return the best matching
	results = [fetch(future) for future in futures]

	best_triangle_count = -Inf
	best_index = -1

	for (i,(triangle_count, gaped_triangles, bipartite_matching_time, scoring_time, U, V)) in enumerate(results)
		if profile
			push!(experiment_profile["matching_timings"],bipartite_matching_time)
			push!(experiment_profile["scoring_timings"],scoring_timings)
		end
		if triangle_count > best_triangle_count
			best_index = i
			best_triangle_count = triangle_count
		end
	end

	(_,_,_,_, best_U, best_V) = results[i]

	best_X = best_U*best_V'


	addprocs(1)
	if profile
		return best_U, best_V, best_triangle_count, experiment_profile
	else
		return best_U, best_V, best_triangle_count
	end

end

function produce_scored_matching(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,
                                 U::Array{Float64,2},V::Array{Float64,2};
								 kwargs...)

    #unsure if low rank option exists for exact solution.
	X= U*V'
	sparseX = sparse(X)
	return TAME_score(A,B,sparseX;kwargs...), U , V

end

#old code
#runs TAME, but reduces down to lowest rank form first
function lowest_rank_TAME_original_svd(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
                          U_0::Array{F,2},V_0::Array{F,2},
                          β::F, max_iter::Int,tol::F,α::F;profile=false) where {F <:AbstractFloat}

	@assert size(U_0,2) == size(V_0,2)

    dimension = minimum((A.n,B.n))

    best_triangle_count = -Inf
    best_x = zeros(size(U_0,1),size(V_0,1))
    best_index = -1


	X_k = U_0 * V_0'
	X_0 = copy(X_k)
#    X_k = copy(W)
 #   X_k_1 = copy(W)
 	rank_0 = size(U_0,2)
	#All_vectors = zeros(A.n,B.n,max_iter)
    i = 1
    lambda = Inf

    if profile
        experiment_profile = Dict(
            "ranks"=>[],
            "contraction_timings"=>[],
            "svd_timings"=>[],
            "matching_timings"=>[],
            "scoring_timings"=>[]
        )
    end

	U_k = copy(U_0)
	V_k = copy(V_0)

    while true

        if profile
            X_k_1,t = @timed kron_contract(A,B,U_k,V_k)
            push!(experiment_profile["contraction_timings"],t)
        else
            X_k_1 = kron_contract(A,B,U_k,V_k)
        end

        new_lambda = dot(X_k_1,X_k)

        if β != 0.0
            X_k_1 .+= β * X_k
        end

        if α != 1.0
            X_k_1 = α * X_k_1 + (1 - α) * X_0
        end

        X_k_1 ./= norm(X_k_1)
		sparse_X_k_1 = sparse(X_k_1)

        if profile
            triangles, gaped_triangles, matching_time, scoring_time = TAME_score(A,B,sparse_X_k_1;return_timings=true)
            push!(experiment_profile["matching_timings"],matching_time)
            push!(experiment_profile["scoring_timings"],scoring_time)
        else
            triangles, gaped_triangles = TAME_score(A,B,sparse_X_k_1)
        end

        println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(gaped_triangles)")

        if triangles > best_triangle_count
            best_x = copy(X_k_1)
            best_triangle_count = triangles
            best_iterate = i
        end

        println("λ: $(new_lambda)")

        if abs(new_lambda - lambda) < tol || i >= max_iter
            if profile
                return best_x, best_triangle_count, experiment_profile
            else
                return best_x, best_triangle_count
            end
        else

			#get the low rank factorization for the next one

			rank_k = size(U_k,2)
			rank_k_1 = rank_k^2 + rank_k + rank_0

			if profile
				(result,t) = @timed svds(sparse_X_k_1,nsv = rank_k_1)
				push!(experiment_profile["svd_timings"],t)
			else
				result = svds(sparse_X_k_1,nsv = rank_k_1)
			end

			U,S,VT = result[1]
			singular_indexes = [i for i in 1:length(S) if S[i] > S[1]*eps(Float64)*dimension]

			println("rank is",length(singular_indexes))
			if profile
				push!(experiment_profile["ranks"],length(singular_indexes))
			end

			U_k = U[:,singular_indexes]
			V_k = VT[:,singular_indexes]*diagm(S[singular_indexes])

			X_k = copy(X_k_1)
			lambda = new_lambda
			i += 1
        end

    end

end


function lowest_rank_TAME_original_svd_for_ranks(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
                          U_0::Array{F,2},V_0::Array{F,2},
                          β::F, max_iter::Int,tol::F,α::F) where {F <:AbstractFloat}

	@assert size(U_0,2) == size(V_0,2)

    dimension = minimum((A.n,B.n))

    best_triangle_count = -Inf
    best_x = zeros(size(U_0,1),size(V_0,1))
    best_index = -1


	X_k = U_0 * V_0'
	X_0 = copy(X_k)
#    X_k = copy(W)
 #   X_k_1 = copy(W)
 	rank_0 = size(U_0,2)
	#All_vectors = zeros(A.n,B.n,max_iter)
    i = 1
    lambda = Inf

	ranks = []

	U_k = copy(U_0)
	V_k = copy(V_0)

    while true

		X_k_1 = kron_contract(A,B,U_k,V_k)

        new_lambda = dot(X_k_1,X_k)

        if β != 0.0
            X_k_1 .+= β * X_k
        end

        if α != 1.0
            X_k_1 = α * X_k_1 + (1 - α) * X_0
        end

        X_k_1 ./= norm(X_k_1)
		sparse_X_k_1 = sparse(X_k_1)

		#println("λ: $(new_lambda)")

        if abs(new_lambda - lambda) < tol || i >= max_iter
    		return best_x, ranks
        else

			#get the low rank factorization for the next one
			rank_k = size(U_k,2)
			rank_k_1 = rank_k^2 + rank_k + rank_0
			result = svds(sparse_X_k_1,nsv = rank_k_1)

			U,S,VT = result[1]
			singular_indexes = [i for i in 1:length(S) if S[i] > S[1]*eps(Float64)*dimension]

			#println("rank is",length(singular_indexes))
			push!(ranks,length(singular_indexes))

			U_k = U[:,singular_indexes]
			V_k = VT[:,singular_indexes]*diagm(S[singular_indexes])

			X_k = copy(X_k_1)
			lambda = new_lambda
			i += 1
        end

    end

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