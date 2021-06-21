
using SparseArrays
using BenchmarkTools
using JLD
include("LambdaTAME.jl")


function find_edge_case(n::Int,p_remove)

	while true
		d = 2
		degreedist=LogNormal(log(4),1.5)
		A = spatial_network(n, d; degreedist=degreedist)
		p = nnz(A)/n^2
		p_add = p*p_remove/(1-p)
		B = ER_noise_model(A,n,p_remove,p_add)

		A_ten = graph_to_ThirdOrderTensor(A)
		B_ten = graph_to_ThirdOrderTensor(B)

		U = rand(A.n,10)
		V = rand(B.n,10)
		X = U*V'
		x = reshape(X,A.n*B.n)
		ABdatasym = setup_tame_data(A_ten,B_ten)
	
		charlie_y = reshape(kron_contract(A_ten,B_ten,U,V),A.n*B.n)
		david_y = impTTVnodesym(A_ten.n,B_ten.n,x,ABdatasym...) 

		diff = norm(david_y - charlie_y)/norm(charlie_y)
		if diff > 1e-10
			return A,B,x
		else 
			println("passed with diff $diff")
		end

	end

end

function test_davids_code(n::Int,p_remove::Float64)

	#Random.seed!(0)

    d = 2
	degreedist=LogNormal(log(4),1.5)
	A = spatial_network(n, d; degreedist=degreedist)
	p = nnz(A)/n^2
	p_add = p*p_remove/(1-p)
	B = ER_noise_model(A,n,p_remove,p_add)
	return test_davids_code(A,B)
end


function test_davids_code(A,B)

	d = 100

	U = rand(size(A,1),d)
	V = rand(size(B,1),d)
	X = U*V'
	x = reshape(X,size(A,1)*size(B,1))
	#x = rand(size(A,1)*size(B,1))

#	ABdata = setup_tame_data(A,B)

	#david_y = impTTVsym(A,B,x,ABdatasym...)

	A_ten = graph_to_ThirdOrderTensor(A)
	B_ten = graph_to_ThirdOrderTensor(B)

	A_U_ten = graph_to_UnweightedThirdOrderTensor(A)
	B_U_ten = graph_to_UnweightedThirdOrderTensor(B)

	ABdatasym = setup_tame_data(A_ten,B_ten)

	println("A tris: $(size(A_ten.indices,1)*6)")
	println("B tris: $(size(B_ten.indices,1)*6)")
	charlie_y = implicit_contraction(A_ten,B_ten,x)


	#println("running david's non-symmetric code...")
	#@btime impTTV($A,$B,$x,$ABdata...) samples=10

	println("running david's symmetric code...")
	david_y = @btime impTTVnodesym($A_ten.n,$B_ten.n,$x,$ABdatasym...) samples =10
	david_y2 = @btime impTTVnodesym($A_U_ten, $B_U_ten,$x) samples =10

	println("Unweighted Ten LowRank Contraction Code")
	#@btime get_kron_contract_comps($A_U_ten,$B_U_ten,$U,$V) samples = 10

	#println("LowRank Contraction Code")
	#@btime get_kron_contract_comps($A_ten,$B_ten,$U,$V) samples = 50
	println("running my code...")


	test_indices = Matrix(A_ten.indices')
	test_indices2 = Matrix(B_ten.indices')
	#charlie_y = @btime implicit_contraction2($test_indices,$test_indices2,$A_ten.n,$B_ten.n,$x)
	println("relative norm difference is : ",norm(david_y - charlie_y)/norm(charlie_y))
	println("relative norm difference is : ",norm(david_y2 - charlie_y)/norm(charlie_y))
	#println("running unweighted Ten code ...")
	#@btime impTTVnodesym($A_U_ten,$B_U_ten,$x) samples = 10
end

function test_row_vs_col(n)

	Random.seed!(0)

    d = 2
	degreedist=LogNormal(log(4),1.5)
	A = spatial_network(n, d; degreedist=degreedist)

	A_ten = graph_to_ThirdOrderTensor(A)

	indices = A_ten.indices
	indices_col= Matrix(A_ten.indices')

	x = rand(A_ten.n)
	y = similar(x)
	y .= 0

	b_row = @btime tensor_vector_contraction_row($A.n,$indices,$x,$y) samples=10
	b_col = @btime tensor_vector_contraction_col($A.n,$indices_col,$x,$y) samples = 10

	return b_row , b_col

end


function tensor_vector_contraction_row(n::Int, indices::Array{Int,2}, x::Array{Float64,1},y)
    @assert length(x) == n


    @inbounds for j=1:size(indices,1)
      #  i_1,i_2,i_3 = indices[j,:]

        y[indices[j,1]]+= 2 * x[indices[j,2]]*x[indices[j,3]]
        y[indices[j,2]] += 2 * x[indices[j,1]]*x[indices[j,3]]
        y[indices[j,3]] += 2 * x[indices[j,1]]*x[indices[j,2]]

    end

    return y
end

function tensor_vector_contraction_col(n::Int, indices::Array{Int,2}, x::Array{Float64,1},y)
    @assert length(x) == n

    @inbounds for (i_1,i_2,i_3) in  eachcol(indices)

        y[i_1] += 2 * x[i_2]*x[i_3]
        y[i_2] += 2 * x[i_1]*x[i_3]
        y[i_3] += 2 * x[i_1]*x[i_2]

    end

    return y
end


function _index_triangles_nodesym(n,Tris::Array{Int,2})

	Ti = [ Vector{Tuple{Int,Int}}(undef, 0) for i in 1:n ]

	for (ti,tj,tk) in eachrow(Tris)
		push!(Ti[ti], (tj,tk))
		push!(Ti[tj], (ti,tk))
		push!(Ti[tk], (ti,tj))
	end
	sort!.(Ti)
	return Ti
end



function setup_tame_data_nodesym(G::SparseMatrixCSC{Int,Int},H::SparseMatrixCSC{Int,Int})
	return _index_triangles_nodesym(G), _index_triangles_nodesym(H)
end



function graph_to_ThirdOrderTensor(A)

	n,n = size(A)

    #build COOTens from graphs
    A_tris = collect(MatrixNetworks.triangles(A))
    A_nnz = length(A_tris)


    A_indices = Array{Int,2}(undef,A_nnz,3)
    for i =1:A_nnz
        A_indices[i,1]= A_tris[i][1]
        A_indices[i,2]= A_tris[i][2]
        A_indices[i,3]= A_tris[i][3]
    end

    A_vals = ones(A_nnz)

    return ThirdOrderSymTensor(n,A_indices,A_vals)

end

function graph_to_UnweightedThirdOrderTensor(A)

	n,n = size(A)

    #build COOTens from graphs
    A_tris = collect(MatrixNetworks.triangles(A))
    A_nnz = length(A_tris)


	Ti = [ Vector{Tuple{Int,Int}}(undef, 0) for i in 1:n ]

	for (ti,tj,tk) in A_tris
		push!(Ti[ti], (tj,tk))
		push!(Ti[tj], (ti,tk))
		push!(Ti[tk], (ti,tj))
	end
	sort!.(Ti)
    return UnweightedThirdOrderSymTensor(n,Ti)

end


function test_contractions(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,
                           U_0::Array{Float64,2},V_0::Array{Float64,2})

	m,d2 = size(U_0)
    n,d1 = size(V_0)

    @assert A.n == m
    @assert B.n == n
	@assert d2 == d1

    X = U_0*V_0'

 	y = implicit_contraction(A,B,reshape(X,n*m))
	U_1, V_1 = get_kron_contract_comps(A,B,U_0,V_0)
	X_2 = U_1*V_1'

	println(norm(X_2  - reshape(y,m,n))/norm(X_2))


end


function test_TAME_iterates(max_iter::Int =  5,TAME_All_vectors=nothing,TAME_All_singvals=nothing,
    						LR_SVD_All_Us=nothing, LR_SVD_All_Vs=nothing,LR_SVD_All_singvals=nothing)

#	A = load_ThirdOrderSymTensor("../data/sparse_tensors/worm_PHY1.ssten")
	B = load_ThirdOrderSymTensor("../data/sparse_tensors/yeast_Y2H1.ssten")	
	A = load_ThirdOrderSymTensor("../data/sparse_tensors/yeast_PHY2.ssten")

	X_0 = ones(A.n,B.n)
	U_0 = ones(A.n,1)
	V_0 = ones(B.n,1)

	X_0 ./= norm(X_0)
	normalization_factor = sqrt(tr((V_0'*V_0)*(U_0'*U_0)))
	U_0 ./= sqrt(normalization_factor)
	V_0 ./= sqrt(normalization_factor)


	β = 1.0
	tol = 1e-6
	α = .85

	if TAME_All_vectors == nothing && TAME_All_singvals == nothing
		println("Running TAME")
		TAME_best_x, TAME_best_triangles, TAME_All_vectors, TAME_All_singvals = TAME_test(A,B,X_0,β,max_iter,tol,α)
	end
	#=
	if LR_SVD_All_Us==nothing && LR_SVD_All_Vs==nothing && LR_SVD_All_singvals==nothing
		println("Running SVD_LR_TAME")
		LR_SVD_TAME_best_x, LR_SVD_TAME_best_triangles, LR_SVD_All_Us, LR_SVD_All_Vs, LR_SVD_All_singvals =
			lowest_rank_TAME_original_svd(A,B,U_0,V_0,β,max_iter,tol,α)
	end
	=#
	println("Running LR TAME")
	LR_TAME_best_triangles, LR_All_Us, LR_All_Vs, LR_All_singvals  =
	     lowest_rank_TAME_test_no_comments(A,B,U_0,V_0,β,max_iter,tol,α,100000)


	for i =1:max_iter-1
		TAME_x = TAME_All_vectors[:,i]

		#LR_SVD_U = LR_SVD_All_Us[i]
		#LR_SVD_V = LR_SVD_All_Vs[i]
		#LR_SVD_TAME_x = reshape(LR_SVD_U*LR_SVD_V',A.n*B.n)

		LR_U = LR_All_Us[i]
		LR_V = LR_All_Vs[i]
		LR_TAME_x = reshape(LR_U*LR_V',A.n*B.n)

		println("iterate $i TAME vs LR  :$(norm(TAME_x - LR_TAME_x)/norm(TAME_x))")
		#println("iterate $i TAME vs SVD :$(norm(TAME_x - LR_SVD_TAME_x)/norm(TAME_x))")
		#println("iterate $i SVD  vs LR  :$(norm(LR_SVD_TAME_x - LR_TAME_x)/norm(LR_TAME_x))")

	#	println("|  ---   Low Rank Forms  ---   |")
	#	println(size(LR_SVD_U))
	#	println(size(LR_U))
	#	println("iterate $i TAME vs SVD U norms:$(norm(LR_SVD_U - LR_U)/norm(LR_SVD_U))")
	#	println("iterate $i SVD  vs LR  V norms:$(norm(LR_SVD_V - LR_V)/norm(LR_SVD_V))")

	end

	println("TAME triangles:$TAME_best_triangles")
	#println("SVD triangles :$LR_SVD_TAME_best_triangles")
	println("LR triangles  :$LR_TAME_best_triangles")

	#return TAME_All_vectors,TAME_All_singvals, LR_SVD_All_Us, LR_SVD_All_Vs,LR_SVD_All_singvals, LR_All_Us, LR_All_Vs, LR_All_singvals
end

function lowest_rank_TAME_test_no_comments(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
                          U_0::Array{F,2},V_0::Array{F,2}, β::F, max_iter::Int,tol::F,α::F,
						  max_rank::Int = minimum((A.n,B.n))) where {F <:AbstractFloat}

	@assert size(U_0,2) == size(V_0,2)

	dimension = minimum((A.n,B.n))

    best_triangle_count = -Inf
	#best_U::Array{Float64,2}
	#best_V::Array{Float64,2}
    best_x = zeros(size(U_0,1),size(V_0,1))
    best_index = -1


	#TODO: test sqrt(tr((V'*V)*(U'*U)))
	#normalization_factor =norm(U_0*V_0')
	normalization_factor = sqrt(tr((V_0'*V_0)*(U_0'*U_0)))
	U_0 ./= sqrt(normalization_factor)
	V_0 ./= sqrt(normalization_factor)

	All_Us = []
	All_Vs = []
	All_singvals = []

	U_k = copy(U_0)
	V_k = copy(V_0)

	i = 1
    lambda = Inf

    for _ in 1:max_iter

		 A_comps, B_comps = get_kron_contract_comps(A,B,U_k,V_k)

		if α != 1.0 && β != 0.0
			U_temp = hcat(sqrt(α) * A_comps, sqrt(α * β) * U_k, sqrt(1-α) * U_0)
			V_temp = hcat(sqrt(α) * B_comps, sqrt(α * β) * V_k, sqrt(1-α) * V_0)
		elseif α != 1.0
			println("made it 1")
			U_temp = hcat(sqrt(α)*A_comps, sqrt(1-α)*U_0)
			V_temp = hcat(sqrt(α)*B_comps, sqrt(1-α)*V_0)
		elseif β != 0.0
			println("made it 2")
			U_temp = hcat(A_comps, sqrt(β) * U_k)
			V_temp = hcat(B_comps, sqrt(β) * V_k)
		else
			U_temp = A_comps
			V_temp = B_comps
		end

#			A_U,A_S,A_Vt = svd(U_temp)
#			B_U,B_S,B_Vt = svd(V_temp)
		A_Q,A_R = qr(U_temp)
		B_Q,B_R = qr(V_temp)

		core = A_R*B_R'
		C_U,C_S,C_Vt = svd(core)

		#singular_indexes= [i for i in 1:1:minimum((max_rank,length(C_S))) if C_S[i] > C_S[1]*eps(Float64)*dimension]
		singular_indexes= [i for i in 1:1:minimum((max_rank,length(C_S))) if C_S[i] > 1e-12]


		U_k_1 = A_Q*C_U[:,singular_indexes]
		V_k_1 = B_Q*(C_Vt[:,singular_indexes]*diagm(C_S[singular_indexes]))

	#	normalization_factor = norm(U_k_1*V_k_1')
	normalization_factor = sqrt(tr((V_k_1'*V_k_1)*(U_k_1'*U_k_1)))
		#println(abs(normalization_factor - norm(U_k_1*V_k_1')))
		U_k_1 ./= sqrt(normalization_factor)
		V_k_1 ./= sqrt(normalization_factor)

	#	println("norm post normalization:",norm(U_k_1*V_k_1'))

		Y, Z = get_kron_contract_comps(A,B,U_k_1,V_k_1)

		lam = tr((Y'*U_k_1)*(V_k_1'*Z))
	#	lam = dot(U_k_1*V_k_1',kron_contract(A,B,U_k_1,V_k_1))
 	#	println("lam diff:$(abs(lam1-lam))")


		#evaluate the matchings
#		sparse_X_k_1 = sparse()
		triangles, gaped_triangles = TAME_score(A,B,U_k_1*V_k_1')
		if triangles > best_triangle_count
			best_triangle_count  = triangles
			best_U = copy(U_k_1)
			best_V = copy(U_k_1)
		end


		println("λ_$i: $(lam) -- rank:$(length(singular_indexes)) -- tris:$(triangles) -- gaped_t:$(gaped_triangles)")

        if abs(lam - lambda) < tol || i >= max_iter
    		#return best_U, best_V, best_triangle_count ,
			return best_triangle_count, All_Us, All_Vs, All_singvals
        end

		#get the low rank factorization for the next one
		U_k = copy(U_k_1)
		V_k = copy(V_k_1)
		push!(All_singvals,C_S)
		push!(All_Us,copy(U_k))
		push!(All_Vs,copy(V_k))

		lambda = lam
		i += 1

    end

end

function LowRankTAME_profiled_RSVD(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
	U_0::Array{F,2},V_0::Array{F,2}, β::F, max_iter::Int,tol::F,α::F;
	max_rank::Int = minimum((A.n,B.n)),update_user::Int=-1,	
	no_matching::Bool=false,low_rank_matching::Bool=false) where {F <:AbstractFloat}

@assert size(U_0,2) == size(V_0,2)

dimension = minimum((A.n,B.n))

best_triangle_count::Int = -1
best_matching = Dict{Int,Int}()
best_x = zeros(size(U_0,1),size(V_0,1))
best_index = -1
triangles = -1
gaped_triangles = -1

normalization_factor = sqrt(tr((V_0'*V_0)*(U_0'*U_0)))
U_0 ./= sqrt(normalization_factor)
V_0 ./= sqrt(normalization_factor)


experiment_profile = Dict{String,Union{Array{F,1},Array{Array{F,1},1}}}(
"ranks"=>Array{Float64,1}(undef,0),
"contraction_timings"=>Array{Float64,1}(undef,0),
"svd_timings"=>Array{Float64,1}(undef,0),
"qr_timings"=>Array{Float64,1}(undef,0),
"matched_tris"=>Array{Float64,1}(undef,0),
"sing_vals"=>Array{Array{Float64,1},1}(undef,0),
"matching_timings"=>Array{Float64,1}(undef,0),
"scoring_timings"=>Array{Float64,1}(undef,0)
)

U_k = copy(U_0)
V_k = copy(V_0)

best_U::Array{F,2} = copy(U_k)
best_V::Array{F,2} = copy(U_k)

lambda = Inf

for i in 1:max_iter


	(A_comps, B_comps),t = @timed get_kron_contract_comps(A,B,U_k,V_k)
	push!(experiment_profile["contraction_timings"],t)

	lambda_k_1 = tr((B_comps'*V_k)*(U_k'*A_comps))


	if α != 1.0 && β != 0.0
	U_temp = hcat(sqrt(α) * A_comps, sqrt(α * β) * U_k, sqrt(1-α) * U_0)
	V_temp = hcat(sqrt(α) * B_comps, sqrt(α * β) * V_k, sqrt(1-α) * V_0)
	elseif α != 1.0
	U_temp = hcat(sqrt(α)*A_comps, sqrt(1-α)*U_0)
	V_temp = hcat(sqrt(α)*B_comps, sqrt(1-α)*V_0)
	elseif β != 0.0
	U_temp = hcat(A_comps, sqrt(β) * U_k)
	V_temp = hcat(B_comps, sqrt(β) * V_k)
	else
	U_temp = A_comps
	V_temp = B_comps
	end

	(A_Q,A_R),t_A = @timed qr(U_temp)
	(B_Q,B_R),t_B = @timed qr(V_temp)
	push!(experiment_profile["qr_timings"],t_A + t_B)
	(A_U,A_S::Array{Float64,1},A_V),t_A = @timed svd(A_R)
	(B_U,B_S::Array{Float64,1},B_V),t_B = @timed svd(B_R)

#	core = A_R*B_R'
#	(C_U,C_S::Array{Float64,1},C_Vt),t = @timed svd(core)
	push!(experiment_profile["svd_timings"],t_A + t_B)


	singular_indexes_A= [i for i in 1:1:minimum((max_rank,length(A_S))) if A_S[i] > A_S[1]*eps(Float64)*dimension]
	singular_indexes_B= [i for i in 1:1:minimum((max_rank,length(B_S))) if B_S[i] > B_S[1]*eps(Float64)*dimension]
	push!(experiment_profile["sing_vals"],C_S)
	push!(experiment_profile["ranks"],float(length(singular_indexes)))

	U_k_1 = A_Q*(A_U[:,singular_indexes_A]*(A_V[:,singular_indexes_A]*diagm(A_S[singular_indexes_A]))')
	V_k_1 = B_Q*(B_U[:,singular_indexes_B]*(B_V[:,singular_indexes_B]*diagm(B_S[singular_indexes_B]))')

	normalization_factor = sqrt(tr((V_k_1'*V_k_1)*(U_k_1'*U_k_1)))

	U_k_1 ./= sqrt(normalization_factor)
	V_k_1 ./= sqrt(normalization_factor)



	if !no_matching
		#evaluate the matchings
		if low_rank_matching
			triangles, gaped_tris, matching, matching_time, scoring_time = TAME_score(A,B,U_k_1,V_k_1;return_timings=true)
		else
			triangles, gaped_tris, matching, matching_time, scoring_time = TAME_score(A,B,U_k_1*V_k_1';return_timings=true)
		end

		push!(experiment_profile["matched_tris"],float(triangles))
		push!(experiment_profile["matching_timings"],float(matching_time))
		push!(experiment_profile["scoring_timings"], float(scoring_time))

		if triangles > best_triangle_count
			best_matching = matching
			best_triangle_count  = triangles
			best_U = copy(U_k_1)
			best_V = copy(V_k_1)
		end
	end

	if update_user != -1 && i % update_user == 0
		println("λ_$i: $(lambda_k_1) -- rank:$(length(singular_indexes)) -- tris:$(triangles) -- gaped_t:$(gaped_tris)")
	end

	if abs(lambda_k_1 - lambda) < tol || i >= max_iter
		#=
		triangles,_= TAME_score(A,B,sparse(best_U*best_V');return_timings=false)
		if triangles > best_triangle_count 
		best_triangle_count = triangles
		end
		=#
		return best_U, best_V, best_triangle_count,best_matching, experiment_profile
	end
	#get the low rank factorization for the next one
	U_k = copy(U_k_1)
	V_k = copy(V_k_1)

	lambda = lambda_k_1
end



return best_U, best_V, best_triangle_count,best_matching, experiment_profile
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


	A_Q,A_R = qr(U_temp)
	B_Q,B_R = qr(V_temp)

	core = A_R*B_R'
	C_U,C_S,C_Vt = svd(core)

	singular_indexes= [i for i in 1:1:minimum((max_rank,length(C_S))) if C_S[i] > C_S[1]*eps(Float64)*dimension]

	U_k_1 = A_Q*C_U[:,singular_indexes]
	V_k_1 = B_Q*(C_Vt[:,singular_indexes]*diagm(C_S[singular_indexes]))

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

    best_triangle_count = -Inf
    best_x = zeros(size(U_0,1),size(V_0,1))
    best_index = -1


	#TODO: test sqrt(tr((V'*V)*(U'*U)))
	#normalization_factor =norm(U_0*V_0')
	normalization_factor = sqrt(tr((V_0'*V_0)*(U_0'*U_0)))
	U_0 ./= sqrt(normalization_factor)
	V_0 ./= sqrt(normalization_factor)

	U_k = copy(U_0)
	V_k = copy(V_0)

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


		 A_comps, B_comps = get_kron_contract_comps(A,B,U_k,V_k)

		if α != 1.0 && β != 0.0
			U_temp = hcat(sqrt(α) * A_comps, sqrt(α * β) * U_k, sqrt(1-α) * U_0)
			V_temp = hcat(sqrt(α) * B_comps, sqrt(α * β) * V_k, sqrt(1-α) * V_0)
		elseif α != 1.0
			println("made it 1")
			U_temp = hcat(sqrt(α)*A_comps, sqrt(1-α)*U_0)
			V_temp = hcat(sqrt(α)*B_comps, sqrt(1-α)*V_0)
		elseif β != 0.0
			println("made it 2")
			U_temp = hcat(A_comps, sqrt(β) * U_k)
			V_temp = hcat(B_comps, sqrt(β) * V_k)
		else
			U_temp = A_comps
			V_temp = B_comps
		end

#			A_U,A_S,A_Vt = svd(U_temp)
#			B_U,B_S,B_Vt = svd(V_temp)
		A_Q,A_R = qr(U_temp)
		B_Q,B_R = qr(V_temp)

		core = A_R*B_R'
		C_U,C_S,C_Vt = svd(core)

		singular_indexes= [i for i in 1:1:minimum((max_rank,length(C_S))) if C_S[i] > C_S[1]*eps(Float64)*dimension]

		U_k_1 = A_Q*C_U[:,singular_indexes]
		V_k_1 = B_Q*(C_Vt[:,singular_indexes]*diagm(C_S[singular_indexes]))

		#C = norm(U_k_1*V_k_1')
		normalization_factor = sqrt(tr((V_k_1'*V_k_1)*(U_k_1'*U_k_1)))
		U_k_1 ./= sqrt(C)
		V_k_1 ./= sqrt(C)
		lam = dot(U_k_1*V_k_1',kron_contract(A,B,U_k_1,V_k_1))

		produce_scored_matching(A,B,U_k_1,V_k_1)

		#evaluate the matchings
		sparse_X_k_1 = sparse(U_k_1*V_k_1')
		triangles, gaped_triangles = TAME_score(A,B,sparse_X_k_1)
		if triangles > best_triangle_count
			best_triangle_count  = triangles
			best_U = copy(U_k_1)
			best_V = copy(U_k_1)
		end


		println("λ_$i: $(lam) -- rank:$(length(singular_indexes)) -- tris:$(triangles) -- gaped_t:$(gaped_triangles)")

        if abs(lam - lambda) < tol || i >= max_iter
    		return best_U, best_V, best_triangle_count
        end

		#get the low rank factorization for the next one
		U_k = copy(U_k_1)
		V_k = copy(V_k_1)

		lambda = lam
		i += 1

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

	All_vectors = zeros(A.n,B.n,max_iter)
	All_Us = []
	All_Vs = []
	All_singvals = []

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
		All_vectors[:,:,i] = X_k_1
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
                return best_x, best_triangle_count, All_Us, All_Vs, All_singvals
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

			push!(All_singvals,S)
			push!(All_Us,U_k)
			push!(All_Vs,V_k)
			X_k = copy(X_k_1)
			lambda = new_lambda
			i += 1
        end

    end

end

#TODO_(7-17-20): need to integrate this propoperly
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


function TAME_test(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,W::Array{F,2},
                       β::F, max_iter::Int,tol::F,α::F;profile=false) where {F <:AbstractFloat}

    dimension = minimum((A.n,B.n))

    if profile
        experiment_profile = Dict(
            "contraction_timings"=>[],
            "matching_timings"=>[],
            "scoring_timings"=>[],
        )
    end

	
	A_Ti, B_Ti = setup_tame_data(A,B)

	best_x = Array{Float64,1}(undef,A.n*B.n)
    best_triangle_count = -Inf
    best_index = -1
    x0 = reshape(W,A.n*B.n)
	x0 ./=norm(x0)
    x_k = copy(x0)

	All_vectors = zeros(A.n*B.n,max_iter)
	All_singvals = []
    i = 1
    lambda = Inf

    while true


        if profile
            x_k_1,t = @timed impTTVnodesym(A.n, B.n, x_k, A_Ti, B_Ti)
            push!(experiment_profile["contraction_timings"],t)
        else
            x_k_1 = impTTVnodesym(A.n, B.n, x_k, A_Ti, B_Ti)
        end

        new_lambda = dot(x_k_1,x_k)

        if β != 0.0
            x_k_1 .+= β * x_k
        end

        if α != 1.0
            x_k_1 = α * x_k_1 + (1 - α) * x0
        end

        x_k_1 ./= norm(x_k_1)

		sparse_X_k_1 = sparse(reshape(x_k_1,A.n,B.n))

        if profile
            triangles, gaped_triangles, bipartite_matching_time, scoring_time = TAME_score(A,B,reshape(x_k_1,A.n,B.n);return_timings=true)
            push!(experiment_profile["matching_timings"],bipartite_matching_time)
            push!(experiment_profile["scoring_timings"],scoring_time)
        else
            triangles, gaped_triangles =  TAME_score(A,B,reshape(x_k_1,A.n,B.n))
        end

        println("finished iterate $(i):tris:$(triangles) -- gaped_t:$(gaped_triangles)")

        if triangles > best_triangle_count
            best_x = copy(x_k_1)
            best_triangle_count = triangles
            best_iterate = i
        end
        println("λ: $(new_lambda)")

        if abs(new_lambda - lambda) < tol || i >= max_iter
            if profile
                return best_x, best_triangle_count, experiment_profile
            else
                return best_x, best_triangle_count , All_vectors ,All_singvals
            end
        else
            x_k = copy(x_k_1)
			All_vectors[:,i] = copy(x_k)
			push!(All_singvals,svdvals(reshape(x_k,(A.n,B.n))))
            lambda = new_lambda
            i += 1
        end

    end


end


#=                                      David's Code                           =#


function _index_triangles(G::SparseMatrixCSC)
	T = collect(triangles(G;symmetries=true))
	Itype = eltype(eltype(T))
	Ti = [ Vector{Tuple{Int,Int}}(undef, 0) for i in 1:size(G,1) ]
	for (ti,tj,tk) in T
		push!(Ti[ti], (tj,tk))
	end
	sort!.(Ti)
	return Ti
end

function setup_tame_data(G::SparseMatrixCSC{Float64,Int},H::SparseMatrixCSC{Float64,Int})
	return _index_triangles(G), _index_triangles(H)
end

function impTTV(G::SparseMatrixCSC{Float64,Int},H::SparseMatrixCSC{Float64,Int},x::Vector{Float64},
		Gti, Hti)
	nG = size(G,1)
	nH = size(H,1)
	X = reshape(x,nG,nH)
	#Xt = copy(X')
	Y = similar(X)
	Y .= 0
	@inbounds for g = 1:nG
		for h = 1:nH
			for (jp,kp) in Hti[h]

				@simd for pair in Gti[g]
					j,k = pair

					@inbounds Y[g,h] += X[j,jp]*X[k,kp]
				end
			end
		end
	end
	y = Y[:]
	return y
end

function _index_triangles_sym(G::SparseMatrixCSC)
	T = collect(triangles(G;symmetries=false))
	Itype = eltype(eltype(T))
	Ti = [ Vector{Tuple{Int,Int}}(undef, 0) for i in 1:size(G,1) ]
	for (ti,tj,tk) in T
		push!(Ti[ti], (tj,tk))
	end
	sort!.(Ti)
	return Ti
end




function _applytri_sym!(Y::Matrix,X::Matrix,i::Integer,j::Integer,k::Integer,
		ip::Integer,jp::Integer,kp::Integer)
	@inbounds begin
		Y[i,ip] += X[j,jp]*X[k,kp]
		Y[i,ip] += X[j,kp]*X[k,jp]
		Y[i,jp] += X[j,ip]*X[k,kp]
		Y[i,jp] += X[j,kp]*X[k,ip]
		Y[i,kp] += X[j,ip]*X[k,jp]
		Y[i,kp] += X[j,jp]*X[k,ip]
		Y[i,ip] += X[k,jp]*X[j,kp]
		Y[i,ip] += X[k,kp]*X[j,jp]
		Y[i,jp] += X[k,ip]*X[j,kp]
		Y[i,jp] += X[k,kp]*X[j,ip]
		Y[i,kp] += X[k,ip]*X[j,jp]
		Y[i,kp] += X[k,jp]*X[j,ip]
		Y[j,ip] += X[i,jp]*X[k,kp]
		Y[j,ip] += X[i,kp]*X[k,jp]
		Y[j,jp] += X[i,ip]*X[k,kp]
		Y[j,jp] += X[i,kp]*X[k,ip]
		Y[j,kp] += X[i,ip]*X[k,jp]
		Y[j,kp] += X[i,jp]*X[k,ip]
		Y[j,ip] += X[k,jp]*X[i,kp]
		Y[j,ip] += X[k,kp]*X[i,jp]
		Y[j,jp] += X[k,ip]*X[i,kp]
		Y[j,jp] += X[k,kp]*X[i,ip]
		Y[j,kp] += X[k,ip]*X[i,jp]
		Y[j,kp] += X[k,jp]*X[i,ip]
		Y[k,ip] += X[i,jp]*X[j,kp]
		Y[k,ip] += X[i,kp]*X[j,jp]
		Y[k,jp] += X[i,ip]*X[j,kp]
		Y[k,jp] += X[i,kp]*X[j,ip]
		Y[k,kp] += X[i,ip]*X[j,jp]
		Y[k,kp] += X[i,jp]*X[j,ip]
		Y[k,ip] += X[j,jp]*X[i,kp]
		Y[k,ip] += X[j,kp]*X[i,jp]
		Y[k,jp] += X[j,ip]*X[i,kp]
		Y[k,jp] += X[j,kp]*X[i,ip]
		Y[k,kp] += X[j,ip]*X[i,jp]
		Y[k,kp] += X[j,jp]*X[i,ip]
	end
end

function impTTVsym(nG::Int,nH::Int,x::Vector{Float64},Gti, Hti)

	X = reshape(x,nG,nH)
	#Xt = copy(X’)
	Y = similar(X)
	Y .= 0

	@inbounds for g = 1:nG
		for h = 1:nH
			for (jp,kp) in Hti[h]
				for (j,k) in Gti[g]
					_applytri_sym!(Y,X,g,j,k,h,jp,kp)
				end
			end
		end
	end
	y = Y[:]
	return y
end

#=
function impTTV(G::SparseMatrixCSC{Float64,Int},H::SparseMatrixCSC{Float64,Int},x::Vector{Float64},
		Gti, Hti)
	nG = size(G,1)
	nH = size(H,1)
	X = reshape(x,nG,nH)
	#Xt = copy(X')
	Y = similar(X)
	Y .= 0
	@inbounds for g = 1:nG
		for h = 1:nH
			for (jp,kp) in Hti[h]
				@simd for pair in Gti[g]
					j,k = pair
					@inbounds Y[g,h] += X[j,jp]*X[k,kp]
				end
			end
		end
	end
	y = Y[:]
	return y
end

=#


#=                                Huda's Code                                =#


function bipartite_matching_setup2(x::Vector{T},ei::Vector{Int64},
                                     ej::Vector{Int64},m::Int64,n::Int64) where T
    (nzi,nzj,nzv) = (ei,ej,x)
    nedges = length(nzi)

    rp = ones(Int64,m+1) # csr matrix with extra edges
    ci = zeros(Int64,nedges+m)
    ai = zeros(Float64,nedges+m)
    tripi = zeros(Int64,nedges+m)
    # 1. build csr representation with a set of extra edges from vertex i to
    # vertex m+i

    rp[1]=0
    for i=1:nedges
        rp[nzi[i]+1]=rp[nzi[i]+1]+1
    end

    rp=cumsum(rp)
    for i=1:nedges
        tripi[rp[nzi[i]]+1]=i
        ai[rp[nzi[i]]+1]=nzv[i]
        ci[rp[nzi[i]]+1]=nzj[i]
        rp[nzi[i]]=rp[nzi[i]]+1
    end

    for i=1:m # add the extra edges
        tripi[rp[i]+1]=-1
        ai[rp[i]+1]=0
        ci[rp[i]+1]=n+i
        rp[i]=rp[i]+1
    end

    # restore the row pointer array
    for i=m:-1:1
        rp[i+1]=rp[i]
    end

    rp[1]=0
    rp=rp.+1

    #check for duplicates in the data
    colind = zeros(Int64,m+n)

    for i=1:m
        for rpi=rp[i]:rp[i+1]-1
            if colind[ci[rpi]] == 1
                error("bipartite_matching:duplicateEdge")
            end
        colind[ci[rpi]]=1
        end

        for rpi=rp[i]:rp[i+1]-1
            colind[ci[rpi]]=0
        end # reset indicator
    end
#    M_setup = Matching_setup(rp,ci,ai,tripi,m,n)
#    return M_setup
	return rp,ci,ai,m,n
	return bipartite_matching_primal_dual(rp,ci,ai,m,n)
end

function bipartite_matching_primal_dual(rp::Vector{Int64}, ci::Vector{Int64},
                    ai::Vector{T}, m::Int64, n::Int64) where T

    # variables used for the primal-dual algorithm
    # normalize ai values # updated on 2-19-2019
    ai ./= maximum(abs.(ai))
    alpha=zeros(Float64,m)
    bt=zeros(Float64,m+n)#beta
    queue=zeros(Int64,m)
    t=zeros(Int64,m+n)
    match1=zeros(Int64,m)
    match2=zeros(Int64,m+n)
    tmod = zeros(Int64,m+n)
    ntmod=0

    # initialize the primal and dual variables
    for i=1:m
        for rpi=rp[i]:rp[i+1]-1
            if ai[rpi] > alpha[i]
               alpha[i]=ai[rpi]
            end
        end
    end

    # dual variables (bt) are initialized to 0 already
    # match1 and match2 are both 0, which indicates no matches

    i=1
    while i<=m
        for j=1:ntmod
            t[tmod[j]]=0
        end #initialize modified map t
        ntmod=0
        # add i to the stack
        head=1
        tail=1
        queue[head]=i
        while head <= tail && match1[i]==0 #queue non-empty + i is unmatched
            k=queue[head]
            for rpi=rp[k]:rp[k+1]-1  #iterate over row k
                j = ci[rpi]
                if ai[rpi] < alpha[k] + bt[j] - 1e-8 
		#			println("$k,$j is tight: $(alpha[k] + bt[j]-ai[rpi])")
                    continue
				end # skip if residual is tight
			
				if t[j]==0  #if neighbor j hasn't been visited
                    tail=tail+1
                    if tail <= m
                        queue[tail]=match2[j]
					end # put whomever j is matched to, on the queue
					
					
                    t[j]=k  
                    ntmod=ntmod+1 
                    tmod[ntmod]=j  #temporarily match j to k, add j to a modified list
                    if match2[j]<1   #if j was previously unmatched 
                        while j>0	 #unfurl all the way to an unmatched node (finding augmenting path)
                            match2[j]=t[j]   
							k=t[j]   
							temp=match1[k]  #store whomever k is matched to for next loop
							match1[k]=j                
							j=temp
							
                        end
                        break #step out to next row 
                    end
                end
            end
            head=head+1
		end
		
        if match1[i] < 1   #if v_i is unmatched, update the flows
            theta=Inf
            for j=1:head-1   #for each element in queue
                t1=queue[j]
                for rpi=rp[t1]:rp[t1+1]-1  #find unmatched neighbor with smallest residual
                    t2=ci[rpi]
                    if t[t2] == 0 && alpha[t1] + bt[t2] - ai[rpi] < theta
                        theta = alpha[t1] + bt[t2] - ai[rpi]
                    end
                end
			end
			#reduce alpha capacity for all in the queue
            for j=1:head-1
                alpha[queue[j]] -= theta
			end
			
			#increase beta flow for all connected to those in the queue
            for j=1:ntmod
                bt[tmod[j]] += theta
            end
            continue
        end
        i=i+1
    end
    val=0
    for i=1:m
        for rpi=rp[i]:rp[i+1]-1
            if ci[rpi]==match1[i]
                val=val+ai[rpi]
            end
        end
    end
    noute = 0
    for i=1:m
        if match1[i]<=n
            noute=noute+1
        end
    end

    #M_output = Matching_output(m,n,val,noute,match1)
    return m,n,val,noute,match1
end



function bipartite_matching_primal_dual_col_based2(X::Adjoint{Float64,Array{T,2}}) where T

	m,n = size(X)

    # variables used for the primal-dual algorithm
    # normalize ai values # updated on 2-19-2019
    X ./= maximum(abs.(X))

    alpha=zeros(Float64,m)
    bt=zeros(Float64,m+n)#beta
    queue=zeros(Int64,m)
    t=zeros(Int64,m+n)
    match1=zeros(Int64,m)
    match2=zeros(Int64,m+n)
    tmod = zeros(Int64,m+n)
    ntmod=0

    # initialize the primal and dual variables

	for j = 1:n
		for i=1:m
			if X[i,j] > alpha[i]
			   alpha[i]=X[i,j]
			end
		end
    end

    # dual variables (bt) are initialized to 0 already
    # match1 and match2 are both 0, which indicates no matches

    i=1
    while i<=m

        for j=1:ntmod
            t[tmod[j]]=0
        end
        ntmod=0
        # add i to the stack
        head=1
        tail=1
        queue[head]=i
        while head <= tail && match1[i]==0 #queue empty + i is unmatched
            k=queue[head]

            for j=1:n #iterate over row k

                if X[k,j] < alpha[k] + bt[j] - 1e-8
                    continue
                end # skip if tight

                if t[j]==0
                    tail=tail+1 #put the potential match in the queue
                    if tail <= m
                        queue[tail]=match2[j]
                    end
                    t[j]=k  #try vertex k for vertex j
                    ntmod=ntmod+1
                    tmod[ntmod]=j
                    if match2[j]<1
                        while j>0
                            match2[j]=t[j]
                            k=t[j]
                            temp=match1[k]
                            match1[k]=j
                            j=temp
                        end
                        break
                    end
                end
            end

			j = k + n #check dummy node
			if 0.0 >= alpha[k] + bt[j] - 1e-8

				if t[j]==0
					tail=tail+1 #put the potential match in the queue
					if tail <= m
						queue[tail]=match2[j]
					end
					t[j]=k  #try vertex k for vertex j
					ntmod=ntmod+1
					tmod[ntmod]=j
					if match2[j]<1
						while j>0
							match2[j]=t[j]
							k=t[j]
							temp=match1[k]
							match1[k]=j
							j=temp
						end
						break
					end
				end
			end
            head=head+1

        end

		if match1[i] < 1  #if node i was unable to be matched, update flows and search for new augmenting path
            theta=Inf
            for j=1:head-1
                t1=queue[j]
                for t2=1:n
                    if t[t2] == 0 && alpha[t1] + bt[t2] - X[t1,t2] < theta
                        theta = alpha[t1] + bt[t2] - X[t1,t2]
                    end
                end
				#check t1's dummy node
				if t[t1 + n] == 0 && alpha[t1] + bt[t1 + n] < theta
                        theta = alpha[t1] + bt[t1 + n]
				end
            end

            for j=1:head-1
                alpha[queue[j]] -= theta
            end
            for j=1:ntmod
                bt[tmod[j]] += theta
            end
            continue
        end

        i=i+1
    end

	#count
    val=0
    for i=1:m
        for j=1:n
            if i==match2[j]
                val=val+X[i,j]
            end
        end
    end

	#count how many are properly matched
    noute = 0
    for j=1:n
        if match2[j]<=m
            noute=noute+1
        end
    end

    #M_output = Matching_output(m,n,val,noute,match1)
	#return val,noute, match2 #convert to original format

	if m > n
		y = [(x > 0) ? x : i + maximum((n,m)) for (i,x) in zip(1:length(match2),match2)][1:n]
	else
		y = [(x > 0) ? x : i + minimum((n,m)) for (i,x) in zip(1:length(match2),match2)][1:n]
	end

    return val,noute,y # #running on adjoint

end


#match1 and match2 correspond to mtaches of left and right nodes respectively. 
#match2 must contain dummy space for dummy nodes. Unmatched nodes are set to 0 for unmatched
function bipartite_matching_primal_dual_col_based(X::Adjoint{T,Array{T,2}},
	                                              match1::Array{Int,1},match2::Array{Int,1}) where T

										
	m,n = size(X)
	@assert length(match1) == m 
	@assert length(match2) == (n+m)


    # variables used for the primal-dual algorithm
    # normalize ai values # updated on 2-19-2019
    X ./= maximum(abs.(X))

    alpha=zeros(Float64,m)
    bt=zeros(Float64,m+n)#beta
    queue=zeros(Int64,m)
    t=zeros(Int64,m+n)
   # match1=zeros(Int64,m)
    #match2=zeros(Int64,m+n)
    tmod = zeros(Int64,m+n)
    ntmod=0

    # initialize the primal and dual variables

	for i = 1:m
		for j=1:n
			if X[i,j] > alpha[i]
			   alpha[i]=X[i,j]
			end
		end
	end

	#add in flow for matches passed in 
	for (i,j) in enumerate(match1)
		if j != 0.0
			alpha[i] -= X[i,j] #double check this is the right orientation
			beta[j] += X[i,j]
		end
	end


    # dual variables (bt) are initialized to 0 already
    # match1 and match2 are both 0, which indicates no matches

    i=1
    while i<=m
    #    println("test $i")
        for j=1:ntmod
            t[tmod[j]]=0
        end
        ntmod=0
        # add i to the stack
        head=1
        tail=1
        queue[head]=i

        while head <= tail && match1[i]==0 #queue empty + i is unmatched

            k=queue[head]
            check_dummy = true
            for j=1:n+1 #iterate over row k
                if j == n+1
                    j= k + n
                end
                if j == k+n #dummy node
                    if 0.0 < alpha[k] + bt[j] - 1e-8
                        continue 
                    end
                else 
                    if X[k,j] < alpha[k] + bt[j] - 1e-8
                        continue
                    end
                end # skip if tight

                if t[j]==0
                    tail=tail+1 #put the potential match in the queue
                    if tail <= m
                        queue[tail]=match2[j]
                    end
                    t[j]=k  #try vertex k for vertex j
                    ntmod = ntmod + 1 
                    tmod[ntmod]=j
                    if match2[j]<1
                        while j>0
                            match2[j]=t[j]
                            k=t[j]
                            temp=match1[k]
                            match1[k]=j
                            j=temp
                        end
                        check_dummy = false
                        break

                    end
                end
            end

            head=head+1


        end

		if match1[i] < 1
            theta=Inf
            for j=1:head-1
                t1=queue[j]
                for t2=1:n
                    if t[t2] == 0 && alpha[t1] + bt[t2] - X[t1,t2] < theta
                        theta = alpha[t1] + bt[t2] - X[t1,t2]
                    end
                end
				#check t1's dummy node
				if t[t1 + n] == 0 && alpha[t1] + bt[t1 + n] < theta
                        theta = alpha[t1] + bt[t1 + n]
				end
            end

            for j=1:head-1
                alpha[queue[j]] -= theta
            end
            for j=1:ntmod
                bt[tmod[j]] += theta
            end
            continue
        end

        i=i+1
    end

	#count


	#count how many are properly matched

    #M_output = Matching_output(m,n,val,noute,match1)
	#return val,noute, match2 #convert to original format

   # println("match1",match1)
    #println("match2",match2)
	if m > n
		y = [(x > 0) ? x : i + maximum((n,m)) for (i,x) in zip(1:length(match2),match2)][1:n]
	else
		y = [(x > 0) ? x : i + minimum((n,m)) for (i,x) in zip(1:length(match2),match2)][1:n]
    end
    
    #compute match value
    val=0.0
    for i=1:m
        for j=1:n
            if i==y[j]
                val=val+X[i,j]
            end
        end
    end

    #compute match cardinality
    noute = 0
    for j=1:n
        if y[j]<=m
            noute=noute+1
        end
    end


    return val,noute,y # #running on adjoint

end

#=
function bipartite_matching_primal_dual(X::Matrix{Float64};tol::Float64=1e-8)
	println("tol is $tol")
    #to get the access pattern right, we must match the right hand side to the left hand side. 

	m,n = size(X)
	@assert m >= n  #error occurs when m < n 

    # variables used for the primal-dual algorithm
    # normalize ai values # updated on 2-19-2019
    #X ./= maximum(abs.(X))

	#code counters

	updated_flows = 0 
	modified_t = 0 
	got_augmented_path = 0 

	#initialize variables
    alpha=zeros(Float64,n)
    bt=zeros(Float64,m+n)#beta
    queue=zeros(Int64,n)
    t=zeros(Int64,m+n)
    match1=zeros(Int64,n)
    match2=zeros(Int64,m+n)
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
   # 	println("test")
        for i=1:ntmod
            t[tmod[i]]=0
        end
        ntmod=0
        # add i to the stack
        head=1
        tail=1
        queue[head]=j
        while head <= tail && match1[j]==0 #queue empty + i is unmatched
 		#	println("test2")
			k=queue[head]
			#println("begining of queue loop")

            for i=1:m+1 #iterate over column k

				if i == m+1 #check the dummy node
					i = k+m
				end
				if i == k+m #dummy nodes don't have weight
					if 0.0 < alpha[k] + bt[i] - 1e-8
						continue
					end
                elseif X[i,k] < alpha[k] + bt[i] - tol
					continue
                end # skip if tight

                if t[i]==0
                    tail=tail+1 #put the potential match in the queue
                    if tail <= m
                        queue[tail]=match2[i]
					end
					modified_t += 1
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
						got_augmented_path += 1
                        break
                    end
                end
            end
            head=head+1

        end

	#	println("queue ",queue[head:tail])
	#	println("t ",t)
		#println("tmod ",tmod)
		#println("match1",match1)
		#println("match2",match2)
		if match1[j] < 1
			updated_flows += 1
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
	
    #M_output = Matching_output(m,n,val,noute,match1)
    return val,noute,match1, match2
end
=#

function main(m,n)

	#bipartite_matching_primal_dual_col_based(rand(3,3)')
	X = rand(m,n)
	#X = [1.0 2.0 -3.0; 4.0 5.0 -6.0 ;7.0 8.0 -9.0]
	#X[1,:] *= -1


	(nzi,nzj,nzv) = findnz(sparse(X))
	#@time MN_result =  bipartite_matching_setup2(nzv,nzi,nzj,maximum(nzi),maximum(nzj))
	MN_result =  bipartite_matching(sparse(X))
	matrix_based_result = bipartite_matching_primal_dual(X,tol=1e-8,normalize_weights=false)
	println(MN_result)
	println(matrix_based_result)

	println("Starting My Code")

	#match1 = zeros(Int,3)
	#match2 = zeros(Int,6)
	#match1[1] = 1
	#match2[1] = 1
	#match1[2] = 2 
	#match2[2] = 2
	#@time LT_result = bipartite_matching_primal_dual_col_based(X',match1,match2)

	#println(MN_result)
	#println(LT_result)
	#println(MN_result2.match == MN_result[end])

#	println(MN_result2.match ==  LT_result[end] )
#	println([x for x in LT_result[end-1]] == [x for x in MN_result[end] if x <= minimum((m,n))])
	#return LT_result[end], MN_result2.match, y

end


function test_matching_use_cases(m,n)

	#=
	println("all ones")
	X = ones(m,n)
	@time MN_result = bipartite_matching(sparse(X))
	@time LT_result = bipartite_matching_primal_dual_col_based(X')
	println(MN_result.match ==  LT_result[end])
	=#
	println("all positive entries")
	X = rand(m,n)

	@time  bipartite_matching(sparse(X))
	@time MN_result = bipartite_matching(sparse(X'))
	@time LT_result = bipartite_matching_primal_dual(X)


	println("match val diff is:",MN_result.weight - LT_result[1])
	println("match cardinalities are the same:",MN_result.cardinality == LT_result[2])
	println("matchines are the same:",MN_result.match ==  LT_result[end])


	println("all positive entries + row of negative")
	X[2,:] *= -1

	@time  bipartite_matching(sparse(X))
	@time MN_result = bipartite_matching(sparse(X))
	@time LT_result = bipartite_matching_primal_dual_col_based(X')

	println(MN_result.cardinality,LT_result[2])

	println("match val diff is:",MN_result.weight - LT_result[1])
	#println("match values are the same:",abs(MN_result.weight - LT_result[1])/LT_result[1] < 1e-15)
	println("match cardinalities are the same:",MN_result.cardinality == LT_result[2])
	println("matchines are the same:",MN_result.match ==  LT_result[end])

	println("all positive entries + col of negative")
	X = rand(m,n)
	X[:,1] *= -1


	@time MN_result =  bipartite_matching(sparse(X))
	@time LT_result = bipartite_matching_primal_dual_col_based(X')
	println("match val diff is:",MN_result.weight - LT_result[1])
	println("match values are the same:",abs(MN_result.weight - LT_result[1])/LT_result[1] < 1e-15)
	println("match cardinalities are the same:",MN_result.cardinality == LT_result[2])
	println("matchines are the same:",MN_result.match ==  LT_result[end])

	println("all negative entries")
	X = -rand(m,n)

	@time MN_result =  bipartite_matching(sparse(X))
	@time LT_result = bipartite_matching_primal_dual_col_based(X')
	println("match val diff is:",MN_result.weight - LT_result[1])
	println("match values are the same:",abs(MN_result.weight - LT_result[1])/LT_result[1] < 1e-15)
	println("match cardinalities are the same:",MN_result.cardinality == LT_result[2])
	println("matchines are the same:",MN_result.match ==  LT_result[end])

	println("all negative entries + row of positive")
	X[2,:] *= -1

	@time MN_result = bipartite_matching(sparse(X))
	@time LT_result = bipartite_matching_primal_dual_col_based(X')
	println("match val diff is:",MN_result.weight - LT_result[1])
	println("match values are the same:",abs(MN_result.weight - LT_result[1])/LT_result[1] < 1e-15)
	println("match cardinalities are the same:",MN_result.cardinality == LT_result[2])
	println("matchines are the same:",MN_result.match ==  LT_result[end])

	println("all negative entries + col of positive")
	X = -rand(m,n)
	X[:,1] *= -1

	@time MN_result =  bipartite_matching(sparse(X))
	@time LT_result = bipartite_matching_primal_dual_col_based(X')
		println("match val diff is:",MN_result.weight - LT_result[1])
	println("match values are the same:",abs(MN_result.weight - LT_result[1])/LT_result[1] < 1e-15)
	println("match cardinalities are the same:",MN_result.cardinality == LT_result[2])
	println("matchines are the same:",MN_result.match ==  LT_result[end])

end

function test_matching_use_cases_btime(m,n)

	println("all positive entries")
	X = rand(m,n)

	@btime MN_result = bipartite_matching(sparse($X)) samples = 10 
	@btime LT_result = bipartite_matching_primal_dual_col_based2($X') samples = 10 

    return 
	#println(MN_result.match ==  LT_result[end])
	#return MN_result.match, LT_result

	println("all positive entries + row of negative")
	X[2,:] *= -1

	@btime MN_result = bipartite_matching(sparse($X))
	@btime LT_result = bipartite_matching_primal_dual_col_based($X')
	#println(MN_result.match==  LT_result[end] )

	println("all positive entries + col of negative")
	X = rand(m,n)
	X[:,1] *= -1

	@btime MN_result =  bipartite_matching(sparse($X))
	@btime LT_result = bipartite_matching_primal_dual_col_based($X')
	#println(MN_result.match ==  LT_result[end] )

	println("all negative entries")
	X = -rand(m,n)

	@btime MN_result =  bipartite_matching(sparse($X))
	@btime LT_result = bipartite_matching_primal_dual_col_based($X')
	#println(MN_result.match ==  LT_result[end] )

	println("all negative entries + row of positive")
	X[2,:] *= -1

	@btime MN_result = bipartite_matching(sparse($X))
	@btime LT_result = bipartite_matching_primal_dual_col_based($X')
	#println(MN_result.match ==  LT_result[end] )

	println("all negative entries + col of positive")
	X = -rand(m,n)
	X[:,1] *= -1

	@btime MN_result =  bipartite_matching(sparse($X))
	@btime LT_result = bipartite_matching_primal_dual_col_based($X')
	#eprintln(MN_result.match ==  LT_result[end] )

end

function estimate_pairwise_alignment_runtimes(files,path,method)

	alphas = [1.0]
	betas = [100.0] #large shift se
	iter = 5
	tol = 1e-6
	no_matching=false
	profile = true

	results = []

	for i in 1:length(files)
        for j in i+1:length(files)
			println(path*files[i])
			_, t = @timed align_tensors(path*files[i],path*files[j];
										method,alphas,betas,iter,tol,no_matching,profile,update_user=1)
										
			println((files[i],files[j],t/iter))
            push!(results,(files[i],files[j],t/iter))
        end
    end

	return results
end



function test_case_run()
	#X_data = JLD.load("/Users/ccolley/PycharmProjects/LambdaTAME/src/testcase.jld")
	#X = X_data["X"]

#	X = rand(3,3)
#	X[1,:]*= -1
	#m,n = size(X)

	m,n = 1000,1000
	d = 50 
	U = rand(m,d)
	V = rand(n,d)
	X = U*V'

	sparse_X = sparse(X)
	ei,ej,x = findnz(sparse_X)
	MN_result, MN_t= @timed bipartite_matching_setup2(x,ei,ej,m,n)
	println("MN time is $MN_t")

	match_dict,_ = low_rank_matching(U,V)
	match1 = zeros(Int,m + n)
	match2 = zeros(Int,n)
	correct = 0
	for (i,j) in match_dict
		match1[i] = j
		match2[j] = i 

		if MN_result[end][i] == j
			correct += 1
		end

	end
	println("number of correct matches: $correct")

	LT_result, LT_t = @timed bipartite_matching_primal_dual_col_based(X',match2,match1)
	println("MN time is $LT_t")
	
	return MN_result, LT_result, X
end

function TAME_score_test(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,U::Array{Float64,2},V::Array{Float64,2};return_timings=false)

	m = size(U,1)
	n = size(V,1)
    if return_timings
		(Match_mapping, _), matching_time = @timed low_rank_matching(U,V)
		
		match1 = zeros(Int,m + n)
		match2 = zeros(Int,n)
	
		for (i,j) in Match_mapping
			match1[i] = j
			match2[j] = i 
		end
		X = U*V'
		_,_,x =bipartite_matching_primal_dual_col_based(X',match2,match1)

		#x = bipartite_matching(X)
        (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,Dict(i => j for (i,j) in enumerate(x)))
        return triangle_count, gaped_triangles, matching_time, scoring_time
    else
        Match_mapping,weight = low_rank_matching(U,V)
        TAME_score(A,B,Match_mapping)
    end

end

function test_low_rank_vs_full_rank_matching(n=100)

	seed = 0 
	p_remove = .01
	d = 10 
	#=
	U = rand(n,d)
	V = rand(n,d)
	X = U*V'

	LR_res,LR_val = low_rank_matching(U,V)
	FR_val,_,_ = bipartite_matching_primal_dual_col_based(X')

	println("LowRank matching result = $LR_val ")
	println("FullRank matching result = $FR_val")
	=#
	#random_graph_exp(n,p_remove,"RandomGeometric",degreedist=LogNormal(log(5),1),seed=seed,profile=true,method="LowRankTAME",alphas=[1.0])

	path = "../data/sparse_tensors/"
	files = [f for f in readdir(path) if !occursin("BioGRID",f)]

	results = []
	for i =1:length(files)
		for j =i+1:length(files)
			X = align_tensors(path*files[i],path*files[j];method="LowRankTAME",
			                  iter=15,tol=1e-6,profile=true)
			push!(results,X)
		end
	end
	
	return results

end

#copied from Experiments.jl file
function setup_random_graph_exp(n::Int, p_remove::Float64,graph_type::String;
	seed=nothing,use_metis=false,degreedist=nothing,perm=nothing,p_edges=nothing,kwargs...)

if seed !== nothing
Random.seed!(seed)
end

if graph_type == "ER"
if p_edges === nothing
	#p = 2*log(n)/n
	p = 20/(2*n) #d_avg = 20 
else
   p = p_edges(n)
end

A = erdos_renyi(n,p)

elseif graph_type == "RandomGeometric"

if degreedist === nothing
	k = 10
	p = k/n
	A = random_geometric_graph(n,k)
else
	d = 2
	A = spatial_network(n, d;degreedist= degreedist)
	p = nnz(A)/n^2
end

elseif graph_type == "HyperKron"
p = .4#2*log(n)/n
r = .4

A = sparse(gpa_graph(n,p,r,5))

else
error("invalid graph type: $graph_type\n must be either 'ER','RandomGeometric' or 'HyperKron'")
end

p_add = p*p_remove/(1-p)
B = ER_noise_model(A,n,p_remove,p_add)

if perm === nothing
	perm = shuffle(1:n)
	#perm = collect(n:-1:1)
end

B = B[perm,perm]

if use_metis
	apply_Metis_permutation!(A,100)
	apply_Metis_permutation!(B,100)
end

return A,B, perm

end

function get_nnz_and_tri_diff(n::Int, p_remove::Float64,graph_type::String;
							  seed=nothing,degreedist=nothing,p_edges=nothing,
							  kwargs...)

	if seed !== nothing
		Random.seed!(seed)
	end
	
	if graph_type == "ER"
		if p_edges === nothing
			#p = 2*log(n)/n
			p = 20/(2*n) #d_avg = 20 
		else
			p = p_edges(n)
		end
	
		A = erdos_renyi(n,p)
		
	elseif graph_type == "RandomGeometric"
	
		if degreedist === nothing
			k = 10
			p = k/n
			A = random_geometric_graph(n,k)
		else
			d = 2
			A = spatial_network(n, d;degreedist= degreedist)
			p = nnz(A)/n^2
		end
	
	elseif graph_type == "HyperKron"
		p = .4#2*log(n)/n
		r = .4
		
		A = sparse(gpa_graph(n,p,r,5))
		
	else
		error("invalid graph type: $graph_type\n must be either 'ER','RandomGeometric' or 'HyperKron'")
	end
	
	p_add = p*p_remove/(1-p)
	B = ER_noise_model(A,n,p_remove,p_add)
		
	n,n = size(B)
	#permute the rows of B
    #perm = shuffle(1:n)
    #B = B[perm,perm]

    #build COOTens from graphs
    A_tris = Set(collect(MatrixNetworks.triangles(A)))
	B_tris = Set(collect(MatrixNetworks.triangles(B)))
	
	tri_diff = 0.0

	tri_diff += length(setdiff(A_tris,B_tris))
	tri_diff += length(setdiff(B_tris,A_tris))

	return norm(A-B,1), tri_diff

end

function get_experiment_difference(p_remove,n_sizes,trial_count,graph_type;kwargs...)

	Random.seed!(0)
	seeds = rand(UInt64, length(p_remove),length(n_sizes), trial_count)

	results = []

	p_index = 1
	for p in p_remove

        n_index = 1

        for n in n_sizes

            for trial = 1:trial_count
  
                seed = seeds[p_index,n_index,trial]

				A_B_nnz_diff, A_ten_B_ten_nnz_diff = get_nnz_and_tri_diff(n, p,graph_type;seed,kwargs...)

				push!(results,(seed,p,n,trial,A_B_nnz_diff,A_ten_B_ten_nnz_diff))
			
			end
			n_index += 1
		end
		p_index += 1
	end


	return results
end


#experimental hot start code
function TAME_score(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,U::Array{Float64,2},V::Array{Float64,2};return_timings=false,use_hot_start = false)

    if return_timings
        if use_hot_start
            println("test")
            (LR_mapping,LR_match_val), LR_matching_time = @timed low_rank_matching(U,V)
            println("Low Rank Matching val $LR_match_val")
            X = U*V'
            match1 = zeros(Int64,size(X,2))
            match2 = zeros(Int64,sum(size(X)))
            for (i,j) in LR_mapping
                match1[j] = i 
                match2[i] = j  
            end
            (match_val,_,matching,_),bipartite_matching_time = @timed bipartite_matching_primal_dual(X;match1=match1,match2=match2)
            println("Matching val $match_val")
            (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
            
            return triangle_count, gaped_triangles, LR_matching_time +bipartite_matching_time, scoring_time, match1, match2, matching
        else
            (Match_mapping, _), matching_time = @timed low_rank_matching(U,V)
            (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,Match_mapping)
            return triangle_count, gaped_triangles, matching_time, scoring_time 
        end
    else
        if use_hot_start
            LR_mapping, _= low_rank_matching(U,V)
            X = U*V'
            match1 = zeros(Int64,size(X,2))
            match2 = zeros(Int64,sum(size(X)))
            for (i,j) in LR_mapping
                match1[j] = i 
                match2[i] = j  
            end
            _,_,matching,_= bipartite_matching_primal_dual(X;match1=match1,match2=match2)
            TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
        else
            Match_mapping,weight = low_rank_matching(U,V)
            TAME_score(A,B,Match_mapping)
        end
    end

end

"""
   Match_mapping is expected to map V_A -> V_B
"""
function TAME_score_test(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,Match_mapping::Dict{Int,Int})
    
    match_len = length(Match_mapping)
    Triangle_check = Dict{Array{Int,1},Int}()
    gaped_triangles = 0
    triangle_count = 0

	matched_triangles = Array{Array{Int,1},1}(undef,0)

    if size(A.indices,1) > size(B.indices,1)

        for i in 1:size(A.indices,1)
            Triangle_check[A.indices[i,:]] = 1
        end

        #invert to map V_B indices to V_A
        Match_mapping = Dict(value => key for (key, value) in Match_mapping)

        for i in 1:size(B.indices,1)
            v_i,v_j,v_k = B.indices[i,:]

            matched_triangle =
              sort([get(Match_mapping,v_i,-1),get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])
    #        println(B.indices[i,:]," -> ",matched_triangle)
            match = get(Triangle_check,matched_triangle,0)
            if match == 1
				triangle_count += 1
				push!(matched_triangles,matched_triangle)
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
				push!(matched_triangles,matched_triangle)
            else
                gaped_triangles += 1
            end
        end
    end

   # triangles, triangle_weight = count_triangles(sub_A,sub_B)

    return triangle_count, gaped_triangles, Match_mapping, matched_triangles #sub_A, sub_B

end

function test_triangle_mappings(n,seed=0)
	p_remove =.01
	graph_type = "RandomGeometric"
	degreedist=LogNormal(5,1)

	A, B = setup_random_graph_exp(n, p_remove,graph_type;seed,degreedist)
	A_ten = graph_to_ThirdOrderTensor(A)
	B_ten = graph_to_ThirdOrderTensor(B)

	println("tensor A triangle count:$(size(A_ten.indices,1))")
	println("tensor B triangle count:$(size(B_ten.indices,1))")

	beta = 1.0
	iter = 3
	tol=1e-6

	best_U, best_V, best_triangle_count, C_matching = LowRankTAME(A_ten,B_ten,ones(A.n,1),ones(B.n,1),beta,iter,tol,1.0;update_user=1)
	X = best_U*best_V'
	MN_output = bipartite_matching(sparse(X)) 
	match_weight,_,C_match, _ = bipartite_matching_primal_dual(X;tol=1e-16,normalize_weights=true)
	println("MN match weight $(MN_output.weight)")
	println("C  match weight $match_weight")

	_,_, MN_mapping, MN_matched_tri = TAME_score_test(A_ten,B_ten,Dict([i=>j for (i,j) in enumerate(MN_output.match)]))
	_,_, C_mapping,  C_matched_tri  = TAME_score_test(A_ten,B_ten,Dict([i=>j for (j,i) in enumerate(C_match)]))
	
	#convert to sets
	MN_mapping = Set(MN_mapping)
	C_mapping = Set(C_mapping)
	MN_matched_tri = Set(MN_matched_tri)
	C_matched_tri  = Set(C_matched_tri)


	println("MN matched tri count: $(length(MN_matched_tri))")
	println("C  matched tri count: $(length(C_matched_tri))")

	difference = symdiff(C_matched_tri,MN_matched_tri)

	MN_diff = intersect(MN_matched_tri,diff)
	C_diff = intersect(C_matched_tri,diff)
	println("|MN_diff| = $(length(MN_diff))")
	println("|C_diff|  = $(length(C_diff))")


	

	map_diff = symdiff(C_mapping,MN_mapping)

	println(length(C_mapping))
	println(length(MN_mapping))
	println(length(map_diff))

	MN_map_diff = intersect(MN_mapping,map_diff)
	C_map_diff = intersect(C_mapping,map_diff)

	MN_diff_edge_weights = Set([X[i,j] for (i,j) in MN_map_diff])
	C_diff_edge_weights =  Set([X[i,j] for (i,j) in C_map_diff])

	println(MN_diff_edge_weights)
	println(C_diff_edge_weights)
	println(symdiff(MN_diff_edge_weights,C_diff_edge_weights))


	MN_mapping, C_mapping
end




#ORB features example


function brute_force_triangles(pts::Array{BitArray{1},1}) where {F <: AbstractFloat}

    n = length(pts)
    # n choose 3 triangles
    #TODO: fix F type error
    triangles = Array{Tuple{Tuple{Int,Int,Int},Tuple{Float64,Float64,Float64}},1}(undef,binomial(n,3))

    index = 1
    for i =1:n
        for j =1:i-1
            for k = 1:j-1
                x = ((k,j,i),vecs_to_triangle(pts[k],pts[j],pts[i]))
                triangles[index] = x
                index += 1
            end
        end
    end

    return triangles

end

function angle(vec1::Union{Array{F,1},Array{Int,1}},vec2::Union{Array{F,1},Array{Int,1}}) where {F <: AbstractFloat}
    acos(dot(vec1, vec2) / (norm(vec1) * norm(vec2))) * 180 / pi
    #TODO: fix DomainError with 1.0000000000000002 problem
end

function vecs_to_triangle(veci::Union{Array{F,1},BitArray{1}},
                          vecj::Union{Array{F,1},BitArray{1}},
                          veck::Union{Array{F,1},BitArray{1}}) where {F <: AbstractFloat}
    seg_ij = vecj - veci
    seg_ik = veck - veci
    seg_jk = veck - vecj

    angle1 = angle(seg_ij, seg_ik)
    angle2 = angle(-seg_ij, seg_jk)
    angle3 = 180.0 - angle1 - angle2
    return (angle1, angle2, angle3)
end

#using GeometricalPredicates, VoronoiDelaunay
function graph_from_knn(key_points::Array{CartesianIndex{2},1},k::Int)

	#find range of the points 
	Is = [x.I[1] for x in key_points]
	Js = [x.I[2] for x in key_points]

	max_I = maximum(Is)
	min_I = minimum(Is)

	max_J = maximum(Js)
	min_J = minimum(Js)

	scaled_J =  [(j - min_J)/max_J + 1.0 for j in Js]
	scaled_I =  [(i - min_I)/max_I + 1.0 for i in Is]

	data = hcat([[x[1],x[2]] for x in zip(scaled_I,scaled_J)]...)
	#=
	points_to_add = [GeometricalPredicates.Point(i,j) for (i,j) in zip(scaled_I,scaled_J)]

	tess = DelaunayTessellation()
	push!(tess, points_to_add)
	=#

	T = BallTree(data)

	# form the edges for sparse
	n = length(key_points)
	ei = Int[]
	ej = Int[]
	for i=1:n
		idxs, dists = knn(T, data[:,i], k+1)
		for j in idxs
			if i != j
				push!(ei,i)
				push!(ej,j)
			end
		end
	end

	A = sparse(ei,ej,1.0,n,n)

	return max.(A,A')

end

function BallTree_triangles()
	T = BallTree(xy)
	# form the edges for sparse
	ei = Int[]
	ej = Int[]
	for i=1:n
		deg = ceil(Int,minimum((rand(degreedist),n-2)))
		idxs, dists = knn(T, xy[:,i], deg+1)
		for j in idxs
		if i != j
			push!(ei,i)
			push!(ej,j)

		end
		end
	end

end

using ImageFeatures, TestImages, Images, ImageDraw, CoordinateTransformations, Rotations
function test_image_features()

	img = testimage("lighthouse")
	img1 = Gray.(img)
	rot = recenter(RotMatrix(5pi/6), [size(img1)...] .÷ 2)  # a rotation around the center
	tform = rot ∘ Translation(-50, -40)
	img2 = warp(img1, tform, axes(img1))

	orb_params = ORB(num_keypoints = 100)

	desc_1, ret_keypoints_1 = create_descriptor(img1, orb_params)
	desc_2, ret_keypoints_2 = create_descriptor(img2, orb_params)

	A = graph_from_knn(ret_keypoints_1,10)
	B = graph_from_knn(ret_keypoints_2,10)

	m,m = size(A)
	n,n = size(B)

    #build COOTens from graphs
    A_tris = collect(MatrixNetworks.triangles(A))
    B_tris = collect(MatrixNetworks.triangles(B))

    A_nnz = length(A_tris)
    B_nnz = length(B_tris)

    A_indices = Array{Int,2}(undef,A_nnz,3)
    B_indices = Array{Int,2}(undef,B_nnz,3)
    for i =1:A_nnz
        A_indices[i,1]= A_tris[i][1]
        A_indices[i,2]= A_tris[i][2]
        A_indices[i,3]= A_tris[i][3]
    end

    for i =1:B_nnz
        B_indices[i,1]= B_tris[i][1]
        B_indices[i,2]= B_tris[i][2]
        B_indices[i,3]= B_tris[i][3]
    end

    A_vals = ones(A_nnz)
    B_vals = ones(B_nnz)

    A_ten = ThirdOrderSymTensor(m,A_indices,A_vals)
    B_ten = ThirdOrderSymTensor(n,B_indices,B_vals)

	U_0 = Matrix(hcat([Array{Float64,1}(x) for x in desc_1]...)')
	V_0 = Matrix(hcat([Array{Float64,1}(x) for x in desc_2]...)')

	X =  LowRankTAME_param_search(A_ten,B_ten;U_0,V_0,betas=[0.0,1.0,10.0,100.0],update_user=1)
	LRT_matchings = [[ret_keypoints_1[i],ret_keypoints_2[j]] for (i,j) in X[5]]


	#matches = match_keypoints(ret_keypoints_1, ret_keypoints_2, desc_1, desc_2, 0.2)
	

	grid = hcat(img1, img2)
	offset = CartesianIndex(0, size(img1, 2))
	map(m -> draw!(grid, LineSegment(m[1], m[2] + offset)), LRT_matchings)
	Images.save("LRT_orb_example.jpg", grid);

end

function test_degree_based_matchings(seed=0,trials::Int=1)

	n=1000
	p_remove = .01
	A,B,perm = setup_random_graph_exp(n, p_remove,"RandomGeometric";
	                             seed,degreedist=LogNormal(log(5),1))

	A_ten = graph_to_ThirdOrderTensor(A)
	B_ten = graph_to_ThirdOrderTensor(B)

	results = []

	for _ in 1:trials
		matching = Dict{Int,Int}(degree_based_matching1(A,B))

		accuracy = sum([1 for (i,j) in enumerate(perm) if get(matching,j,-1) == i])/n

		triangle_count, gaped_triangles, _ = TAME_score(A_ten,B_ten,matching) 
		push!(results,(accuracy,triangle_count))
	end

	return results , min(size(A_ten.indices,1),size(B_ten.indices,1))
end




function degree_based_matching(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int}) where T

	m = size(A,1)
	n = size(B,1)

	d_A = A*ones(n)
	d_B = B*ones(m)

	noise_factor = .01
	p_A = sort(1:m,by=i->d_A[i]+noise_factor*rand()) #breaks ties with some variation
	p_B = sort(1:n,by=j->d_B[j]+noise_factor*rand())

	return collect(zip(p_A,p_B))

end

function lowRankEigenAlign_recovery_exps(n,p_remove)

	trials = 10
	randomized_results = []
	rev_sorted_results = []
	#run randomized 
	for trial in 1:trials
		A,B,perm = setup_random_graph_exp(n, p_remove,"ER";seed=trial)

		iters = 10
		ma,mb,_,_ = align_networks_eigenalign(A,B,iters,"lowrank_svd_union",3)


		matching = Dict{Int,Int}(zip(mb,ma))
		accuracy = sum([1 for (i,j) in enumerate(perm) if get(matching,i,-1) == j])/n

		push!(randomized_results,accuracy)
		println(accuracy)
	end

	for trial in 1:trials
		A,B,perm = setup_random_graph_exp(n, p_remove,"ER";seed=trial,perm=collect(n:-1:1))

		iters = 10
		ma,mb,_,_ = align_networks_eigenalign(A,B,iters,"lowrank_svd_union",3)

		matching = Dict{Int,Int}(zip(ma,mb))
		accuracy = sum([1 for (i,j) in enumerate(perm) if get(matching,i,-1) == j])/n

		push!(rev_sorted_results,accuracy)
		println(accuracy)
	end
	collect(n:-1:1)

   return randomized_results, rev_sorted_results
end


using Random
using SparseArrays
function embedded_experiment()

	A = sparse(MatrixNetworks.readSMAT("data/sparse_matrices/LVGNA/yeast_Y2H1.smat"))
	n_s = Int(ceil(size(A,1)*1.1))
	embedded_A = sparse(MatrixNetworks.erdos_renyi_undirected(n_s,log(n_s)/n_s))
	embedded_A[1:size(A,1),1:size(A,1)] = A

	perm = collect(1:size(embedded_A,1))
	shuffle!(perm)

	embedded_A = embedded_A[perm,perm]
	dropzeros!(embedded_A)

	accuracy = sum([1 for (i,j) in X[end][end] if perm[i] == j])/n	



end
