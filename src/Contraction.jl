
"""-----------------------------------------------------------------------------
    contracts a tensor represented with indices + values in a symmetric COO
  format. Rows of the indices correspond to sorted indices of a symmetric
  hyper-edge. Values correspond to the edge weights, and v is vector being
  contracted with the
  -----------------------------------------------------------------------------"""
function tensor_vector_contraction(A::ThirdOrderSymTensor, x::Array{Float64,1})
    @assert length(x) == A.n

    y = zeros(Float64,A.n)

	j =1
    @inbounds for (i_1,i_2,i_3) in eachcol(A.indices)

        y[i_1] += 2 * x[i_2]*x[i_3]*A.values[j]
        y[i_2] += 2 * x[i_1]*x[i_3]*A.values[j]
        y[i_3] += 2 * x[i_1]*x[i_2]*A.values[j]

		j +=1
    end

    return y
end


function tensor_vector_contraction(A::UnweightedThirdOrderSymTensor, x::Array{Float64,1})
    @assert length(x) == A.n

    y = zeros(Float64,A.n)

    @inbounds for (i_1,i_2,i_3) in eachcol(A.indices)

        y[i_1] += 2 * x[i_2]*x[i_3]
        y[i_2] += 2 * x[i_1]*x[i_3]
        y[i_3] += 2 * x[i_1]*x[i_2]

    end

    return y
end




"""-----------------------------------------------------------------------------
    contracts a tensor represented with indices + values in a symmetric COO
  format with a vector to return a sparse matrix.
-----------------------------------------------------------------------------"""
function tri_sub_tensor(A::ThirdOrderSymTensor,x::Array{Float64,1})

	index = 0
	nnzs = size(A.indices,2)
	I = zeros(Int64,6*nnzs)
	J = zeros(Int64,6*nnzs)
	V = zeros(Float64,6*nnzs)

    @inbounds for nnz_index = 1:nnzs
		index += 6
		i,j,k = A.indices[:,nnz_index]
		val = A.values[nnz_index]

		I[index-5] = i
		J[index-5] = j
		I[index-4] = j
		J[index-4] = i
		V[index-5] = val*x[k]
		V[index-4] = val*x[k]

		I[index-3] = i
		J[index-3] = k
		I[index-2] = k
		J[index-2] = i
		V[index-3] = val*x[j]
		V[index-2] = val*x[j]

		I[index-1] = j
		J[index-1] = k
		I[index] = k
		J[index] = j
		V[index-1] = val*x[i]
		V[index] = val*x[i]

	end
	return sparse(I,J,V,A.n,A.n)
end


"""-----------------------------------------------------------------------------
    contracts a tensor represented with indices + values in a symmetric COO
  format with a vector to return a sparse matrix.
-----------------------------------------------------------------------------"""
function tri_sub_tensor(A::ThirdOrderSymTensor,x::Array{Float64,1},
    					I::Array{Int64,1},J::Array{Int64,1},V::Array{Float64,1})

	index = 0
	nnzs = size(A.indices,2)
    for nnz_index = 1:nnzs
		index += 6
		i,j,k = A.indices[:,nnz_index]
		val = A.values[nnz_index]

		I[index-5] = i
		J[index-5] = j
		I[index-4] = j
		J[index-4] = i
		V[index-5] = val*x[k]
		V[index-4] = val*x[k]

		I[index-3] = i
		J[index-3] = k
		I[index-2] = k
		J[index-2] = i
		V[index-3] = val*x[j]
		V[index-2] = val*x[j]

		I[index-1] = j
		J[index-1] = k
		I[index] = k
		J[index] = j
		V[index-1] = val*x[i]
		V[index] = val*x[i]

	end
	return sparse(I,J,V,A.n,A.n)
end


"""-----------------------------------------------------------------------------
    contracts a tensor represented with indices + values in a symmetric COO
  format with a vector to return a sparse matrix.
-----------------------------------------------------------------------------"""
function tri_sub_tensor(A::UnweightedThirdOrderSymTensor,x::Array{Float64,1})

	index = 0
	nnzs = 2*sum([length(x) for x in A_U_ten.indices])
	I = zeros(Int64,2*nnzs)
	J = zeros(Int64,2*nnzs)
	V = zeros(Float64,2*nnzs)

	for i in 1:A.n
		for (j,k) in A.indices[i]
			index += 2

			I[index-1] = j
			J[index-1] = k
			I[index] = k
			J[index] = j
			V[index-1] = x[i]
			V[index] = x[i]
		end
	end
	return sparse(I[1:index],J[1:index],V[1:index],A.n,A.n)
end




function kron_contract(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,
                       X::SparseMatrixCSC{Float64,Int64},rank::Int=minimum(size(X)))

	(U,S,VT),_ = svds(X,nsv = rank)
	println("input rank: ",rank)
	println("output rank:",length(S))


	#singular_indexes = [i for i in 1:length(S) if S[i] > S[1]*eps(Float64)*minimum(size(X))]

	#println("rank in kron contract is",length(singular_indexes))

    kron_contract(A,B,U,VT*diagm(S))
end

function kron_contract(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,
                       U::Array{Float64,2},V::Array{Float64,2})
    n,d1 = size(U)
    m,d2 = size(V)
    @assert d1 == d2
    @assert A.n == n
    @assert B.n == m


	#preallocate memory used for building sparse matrices
	A_nnzs = size(A.indices,2)
	A_I = zeros(Int64,6*A_nnzs)
	A_J = zeros(Int64,6*A_nnzs)
	A_V = zeros(Float64,6*A_nnzs)


	B_nnzs = size(B.indices,1)
	B_I = zeros(Int64,6*B_nnzs)
	B_J = zeros(Int64,6*B_nnzs)
	B_V = zeros(Float64,6*B_nnzs)

    result = zeros(A.n * B.n)


    @inbounds for i in 1:d1

	#	sub_A_i = tri_sub_tensor(A,U[:,i])
	#	sub_B_i = tri_sub_tensor(B,V[:,i])
        sub_A_i = tri_sub_tensor(A,U[:,i],A_I,A_J,A_V)
        sub_B_i = tri_sub_tensor(B,V[:,i],B_I,B_J,B_V)
        for j in 1:i

            A_update = (sub_A_i*U[:,j])
            B_update = (sub_B_i*V[:,j])

            if i == j
               result += kron(B_update,A_update)
            else
                result += kron(2*B_update,A_update)
            end
        end
    end

    return reshape(result, A.n,B.n)
end


function get_kron_contract_comps(A::Union{ThirdOrderSymTensor,UnweightedThirdOrderSymTensor},
    							 B::Union{ThirdOrderSymTensor,UnweightedThirdOrderSymTensor},
              			         U::Array{Float64,2},V::Array{Float64,2})

    n,d1 = size(U)
    m,d2 = size(V)
    @assert d1 == d2
    @assert A.n == n
    @assert B.n == m

#    result = zeros(A.n * B.n)

    ileave = (i,j) -> i + A.n*(j-1)

    max_rank = Int((d1+1)*d1/2)

	A_comps = zeros(n,max_rank)
	B_comps = zeros(m,max_rank)
	index = 1

	A_nnzs = size(A.indices,2)
	A_I = zeros(Int64,6*A_nnzs)
	A_J = zeros(Int64,6*A_nnzs)
	A_V = zeros(Float64,6*A_nnzs)


	B_nnzs = size(B.indices,2)
	B_I = zeros(Int64,6*B_nnzs)
	B_J = zeros(Int64,6*B_nnzs)
	B_V = zeros(Float64,6*B_nnzs)


    @inbounds for i in 1:d1


        sub_A_i = tri_sub_tensor(A,U[:,i],A_I,A_J,A_V)
        sub_B_i = tri_sub_tensor(B,V[:,i],B_I,B_J,B_V)

        for j in 1:i

			if i == j
				A_comps[:,index] = (sub_A_i*U[:,j])
				B_comps[:,index] = (sub_B_i*V[:,j])
			else

				A_comps[:,index] = sqrt(2)*(sub_A_i*U[:,j])
				B_comps[:,index] = sqrt(2)*(sub_B_i*V[:,j])
			end
			index += 1
        end
    end

    return A_comps, B_comps
end

function get_kron_contract_comps(A::Union{SymTensor{M,T},SymTensorUnweighted{M}},
						B::Union{SymTensor{M,T},SymTensorUnweighted{M}},
						U::Matrix{T},V::Matrix{T}) where {M <: Motif,T}
	# adapted from contract_all_unique_permutations routine in DistributedTensorConstruction.jl
	#=assuming that 
		A.order == B.order,
		size(U,2) == size(V,2), 
		size(U,1) = A.n, and 
		size(V,1) = B.n
	=#
    m,d = size(U)
    
    A_contraction_components = Array{T,2}(undef,A.n,binomial(d + A.order-2, A.order-1))
	B_contraction_components = Array{T,2}(undef,B.n,binomial(d + B.order-2, B.order-1))
                                                # n choose k w/ replacement
    get_kron_contract_comps!(A,B,U,V,A_contraction_components,B_contraction_components,0,size(U,2),Array{Int}(undef,0))
    
    return A_contraction_components, B_contraction_components
end

function get_kron_contract_comps!(A::Union{SymTensor{M,T},SymTensorUnweighted{M}},
						 B::Union{SymTensor{M,T},SymTensorUnweighted{M}},
						 U::Matrix{T},V::Matrix{T},
						 A_contraction_components::Matrix{T},
						 B_contraction_components::Matrix{T},
						 offset::Int,end_idx::Int,
						 prefix_indices::Vector{Int},) where {M <: Motif,T}


    d = size(U,2)
    if A.order == 3 
        indices = Array{Int}(undef,length(prefix_indices)+2)
        indices[1:end-2] = prefix_indices

        for i = 1:end_idx
            indices[end-1] = i  
            sub_A_mat = contract_to_mat(A, U[:,i])
			sub_B_mat = contract_to_mat(B, V[:,i])
            #for j = i:size(U,2)
            for j = 1:i
                indices[end] = j

                factor = DistributedTensorConstruction.compute_multinomial(indices)
                #println("edge:$indices  factor:$factor  ")

                idx = offset + i*(i-1)÷2 + j
            
                A_contraction_components[:,idx] = factor*(sub_A_mat*U[:,j])
				B_contraction_components[:,idx] = (sub_B_mat*V[:,j])
            end
        end
    else

        running_offset = copy(offset)
        for i = 1:end_idx
            sub_A = single_mode_ttv(A,U[:,i])
			sub_B = single_mode_ttv(B,V[:,i])
            prefix = copy(prefix_indices)
            push!(prefix,i)
     
            get_kron_contract_comps!(sub_A,sub_B, U,V, 
							A_contraction_components,B_contraction_components,
							running_offset,i,prefix) 

            running_offset += binomial(i + A.order-3, A.order-2)
        end

    end

end

function get_kron_contract_comps_with_accumulation_param(A::Union{SymTensor{M,T},SymTensorUnweighted{M}},
						B::Union{SymTensor{M,T},SymTensorUnweighted{M}},
						U::Matrix{T},V::Matrix{T}) where {M <: Motif,T}
	# adapted from contract_all_unique_permutations routine in DistributedTensorConstruction.jl
	#=assuming that 
		A.order == B.order,
		size(U,2) == size(V,2), 
		size(U,1) = A.n, and 
		size(V,1) = B.n
	=#
    X = zeros(T,A.n,B.n)
                                                # n choose k w/ replacement
    get_kron_contract_comps!(A,B,U,V,X,size(U,2),Array{Int}(undef,0))
    
    return X
end

function get_kron_contract_comps!(A::Union{SymTensor{M,T},SymTensorUnweighted{M}},
								B::Union{SymTensor{M,T},SymTensorUnweighted{M}},
								U::Matrix{T},V::Matrix{T},
								X::Matrix{T},end_idx::Int,
								prefix_indices::Vector{Int}) where {M <: Motif,T}


    d = size(U,2)
    if A.order == 3 
        indices = Array{Int}(undef,length(prefix_indices)+2)
        indices[1:end-2] = prefix_indices

        for i = 1:end_idx
            indices[end-1] = i  
            sub_A_mat = contract_to_mat(A, U[:,i])
			sub_B_mat = contract_to_mat(B, V[:,i])
            #for j = i:size(U,2)
            for j = 1:i
                indices[end] = j
                factor = DistributedTensorConstruction.compute_multinomial(indices)

				u = factor*sub_A_mat*U[:,j]
				v = sub_B_mat*V[:,j]

				for col in 1:B.n
					for row in 1:A.n
						X[row,col] += u[row]*v[col]
					end
				end
            end
        end
    else

        #running_offset = copy(offset)
        for i = 1:end_idx
            sub_A = single_mode_ttv(A,U[:,i])
			sub_B = single_mode_ttv(B,V[:,i])
            prefix = copy(prefix_indices)
            push!(prefix,i)
     
            get_kron_contract_comps!(sub_A, sub_B, U, V, X, i, prefix) 

        end

    end

end






function implicit_contraction(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,x::Array{Float64,1})

	error("Code is Broken, must be fixed")
    @assert length(x) == A.n*B.n
    m = A.n
    n = B.n
	X = reshape(x,A.n,B.n)
    Y = similar(X)

    #ileave = (i,j) -> i + m*(j-1)
	i = 1
	j = 1
    @inbounds for (i_1,i_2,i_3) in eachcol(A.indices)

		A_val = A.values[i]
		i += 1
		j = 1
#        for (i_1,i_2,i_3) in permutations(A.indices[i,:])
		for (j_1,j_2,j_3) in eachcol(B.indices)


			#i_1,i_2,i_3
			@inbounds Y[i_1,j_1] += 2*A_val*B.values[j]*X[i_2,j_2]*X[i_3,j_3]
			@inbounds Y[i_2,j_2] += 2*A_val*B.values[j]*X[i_1,j_1]*X[i_3,j_3]
			@inbounds Y[i_3,j_3] += 2*A_val*B.values[j]*X[i_1,j_1]*X[i_2,j_2]

			#i_1,i_3,i_2
			@inbounds Y[i_1,j_1] += 2*A_val*B.values[j]*X[i_3,j_2]*X[i_2,j_3]
			@inbounds Y[i_3,j_2] += 2*A_val*B.values[j]*X[i_1,j_1]*X[i_2,j_3]
			@inbounds Y[i_2,j_3] += 2*A_val*B.values[j]*X[i_1,j_1]*X[i_3,j_2]

			#i_2,i_1,i_3
			@inbounds Y[i_2,j_1] += 2*A_val*B.values[j]*X[i_1,j_2]*X[i_3,j_3]
			@inbounds Y[i_1,j_2] += 2*A_val*B.values[j]*X[i_2,j_1]*X[i_3,j_3]
			@inbounds Y[i_3,j_3] += 2*A_val*B.values[j]*X[i_2,j_1]*X[i_1,j_2]

			#i_2,i_3,i_1
			@inbounds Y[i_2,j_1] += 2*A_val*B.values[j]*X[i_3,j_2]*X[i_1,j_3]
			@inbounds Y[i_3,j_2] += 2*A_val*B.values[j]*X[i_2,j_1]*X[i_1,j_3]
			@inbounds Y[i_1,j_3] += 2*A_val*B.values[j]*X[i_2,j_1]*X[i_3,j_2]

			#i_3,i_2,i_1
			@inbounds Y[i_3,j_1] += 2*A_val*B.values[j]*X[i_2,j_2]*X[i_1,j_3]
			@inbounds Y[i_2,j_2] += 2*A_val*B.values[j]*X[i_3,j_1]*X[i_1,j_3]
			@inbounds Y[i_1,j_3] += 2*A_val*B.values[j]*X[i_3,j_1]*X[i_2,j_2]

			#i_3,i_1,i_2
			@inbounds Y[i_3,j_1] += 2*A_val*B.values[j]*X[i_1,j_2]*X[i_2,j_3]
			@inbounds Y[i_1,j_2] += 2*A_val*B.values[j]*X[i_3,j_1]*X[i_2,j_3]
			@inbounds Y[i_2,j_3] += 2*A_val*B.values[j]*X[i_3,j_1]*X[i_1,j_2]

		end
		j += 1
  #      end
    end

    return Y[:]
end


function impTTVnodesym(A::UnweightedThirdOrderSymTensor,
                       B::UnweightedThirdOrderSymTensor,x::Vector{Float64})

	X = reshape(x,A.n,B.n)
	Y = similar(X)
	Y .= 0
	@inbounds for a = 1:A.n
		for b = 1:B.n
			newval = 0.0
			for (jp,kp) in A.indices[a]
				@simd for pair in B.indices[b]
					j,k = pair
					@inbounds newval += X[jp,j]*X[kp,k]+X[kp,j]*X[jp,k]
				end
			end
			Y[a,b] = 2*newval
		end
	end
	y = Y[:]
	return y

end


function impTTVnodesym(nG::Int,nH::Int,x::Vector{Float64},Gti, Hti)

	X = reshape(x,nG,nH)
	#Xt = copy(X')
	Y = similar(X)
	Y .= 0
	@inbounds for g = 1:nG
		for h = 1:nH
			newval = 0.0
			for (jp,kp) in Gti[g]
				@simd for pair in Hti[h]
					j,k = pair
					@inbounds newval += X[jp,j]*X[kp,k]+X[kp,j]*X[jp,k]
				end
			end
			Y[g,h] = 2*newval
		end
	end
	y = Y[:]
	return y
end

function impTTVnodesym(m::Int,n::Int,order::Int,X::Matrix{T},A_Mi,B_Mi) where T


    Y = zeros(T,size(X)...)

	factor = factorial(order-1)

	permutation_template = collect(permutations(1:(order-1)))

	@inbounds for a = 1:m
		for b = 1:n
			newval = 0.0
			for sub_edge_A in A_Mi[a]
				for sub_edge_B in B_Mi[b]
					
					#_tame_edge_contract!(sub_edge_A, sub_edge_B,newval)
					#for (sub_edge_A_p,sub_edge_B_p) in zip(permutations(sub_edge_A),permutations(sub_edge_B))
					for p in permutation_template

						x_prod = X[sub_edge_A[p[1]],sub_edge_B[1]]
						for i = 2:(order-1)
							x_prod *= X[sub_edge_A[p[i]],sub_edge_B[i]]
						end
						newval += x_prod 

					end
				end
			end
			Y[a,b] = factor*newval
		end
	end

    return Y
end




"""-----------------------------------------------------------------------------
                         Iterative method primatives
-----------------------------------------------------------------------------"""
function SSHOPM_sample_return_all(A::Union{ThirdOrderSymTensor,SymTensorUnweighted}, samples::Int, β::Float64, max_iter::Int, tol::Float64)

	#X = rand(A.n,samples)
	X = rand(Uniform(-1,1),A.n,samples)

	V = Array{Float64,2}(undef,A.n,samples)
	Λ = Array{Float64,1}(undef,samples)

	for i=1:samples
		vec, val = SSHOPM(A,β,max_iter,tol,X[:,i])
		V[:,i] .= vec
		Λ[i] = val
	end
	
	V,Λ

end

function SSHOPM_sample(A::ThirdOrderSymTensor, samples::Int, β::Float64, max_iter::Int, tol::Float64,kwargs...)

	#X = rand(A.n,samples)
	#X = rand(Uniform(-1,1),A.n,ssample)


	argmax_vec = Array{Float64,1}(undef,A.n)
	argmax_val::Float64 = 0.0
	#Vecs = Array{Float64,3}(undef,A.n,B.n,samples)
	Λ = Array{Float64,1}(undef,samples)

	for i=1:samples
		#U,V, val
		vec, val =  SSHOPM(A,β,max_iter,tol,rand(Uniform(-1,1),A.n))

		if abs(val) > abs(argmax_val) 
			argmax_vec = vec
			argmax_val = val
		end
		
		Λ[i] = val
	end
	
	argmax_vec,argmax_val,Λ

end


#TODO: move to appropriate file 
function SSHOPM(A::ThirdOrderSymTensor,β::Float64, max_iter::Int, tol::Float64,
	            x_0::Array{Float64,1}=ones(A.n);update_user::Int=-1)

	#normalize init vector
	x_k = copy(x_0)
	x_k ./=norm(x_0)

    lambda_k = Inf
    i = 1

    while true

		x_k_1 = tensor_vector_contraction(A,x_k)
		lambda_k_1 = x_k_1'*x_k


		if β != 0.0
			x_k_1 .+= β*x_k
        end

		x_k_1 ./= norm(x_k_1)

		res = abs(lambda_k_1 - lambda_k)
		if update_user != -1 && i % update_user == 0
			println("iteration $(i)    λ: $(lambda_k_1)   |λ_k_1 - λ_k| = $res")
		end

        if res < tol || i >= max_iter
            return x_k_1, lambda_k_1
        else
			lambda_k = copy(lambda_k_1)
			x_k = copy(x_k_1)
            i += 1
        end

    end

end

function SSHOPM(A::SymTensorUnweighted,β::Float64, max_iter::Int, tol::Float64,
	x_0::Array{Float64,1}=ones(A.n);update_user::Int=-1)

	#normalize init vector
	buf = zeros(Float64,A.n)
	x_k_1 = Array{Float64,1}(undef,A.n)
	x_k = copy(x_0)
	x_k ./=norm(x_0)

	lambda_k = Inf
	i = 1

	while true

		DistributedTensorConstruction.contraction!(A,x_k,buf)
		x_k_1 .= buf 
		lambda_k_1 = x_k_1'*x_k


		if β != 0.0
			x_k_1 .+= β*x_k
		end

		x_k_1 ./= norm(x_k_1)

		res = abs(lambda_k_1 - lambda_k)
		if update_user != -1 && i % update_user == 0
			println("iteration $(i)    λ: $(lambda_k_1)   |λ_k_1 - λ_k| = $res")
		end

		if res < tol || i >= max_iter
			return x_k_1, lambda_k_1
		else
			lambda_k = copy(lambda_k_1)
			x_k = copy(x_k_1)
			i += 1
			buf .= 0.0
		end

	end

end


function SSHOPM_sample_return_all(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor, samples::Int, β::Float64, max_iter::Int, tol::Float64,kwargs...)

	#X = rand(A.n,samples)
	X = rand(Uniform(-1,1),A.n,B.n,samples)

	Vecs = Array{Float64,3}(undef,A.n,B.n,samples)
	Λ = Array{Float64,1}(undef,samples)

	for i=1:samples
		#U,V, val
		U,V, val = SSHOPM(A,B,X[:,:,i],β,max_iter,tol;kwargs...)
		Vecs[:,:,i] = U*V'
		Λ[i] = val
	end
	
	Vecs,Λ

end

function SSHOPM_sample(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor, samples::Int, β::Float64, max_iter::Int, tol::Float64,kwargs...)

	#X = rand(A.n,samples)
	#X = rand(Uniform(-1,1),A.n,B.n,samples)


	argmax_vec = Array{Float64,1}(undef,A.n*B.n)
	argmax_val::Float64 = 0.0
	#Vecs = Array{Float64,3}(undef,A.n,B.n,samples)
	Λ = Array{Float64,1}(undef,samples)

	for i=1:samples
		#U,V, val
		vec, val = SSHOPM(A,B,rand(Uniform(-1,1),A.n,B.n),β,max_iter,tol;kwargs...)

		if abs(val) > abs(argmax_val) 
			argmax_vec = vec
			argmax_val = val
		end
		
		Λ[i] = val
	end
	
	reshape(argmax_vec,A.n,B.n),argmax_val,Λ

end

#=
function SSHOPM(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,W::Array{F,2},
				β::F, max_iter::Int,tol::F;
				max_rank::Int = minimum((A.n,B.n)),kwargs...) where {F <:AbstractFloat}

	dimension = minimum((A.n,B.n))

	U_k,S,VT =svd(W)
	singular_indexes = [i for i in 1:minimum((max_rank,length(S))) if S[i] > S[1]*eps(Float64)*dimension]

	U = U_k[:,singular_indexes]
	V = VT[:,singular_indexes]*diagm(S[singular_indexes])

	return SSHOPM(A,B,U,V,β,max_iter,tol;kwargs...)
end
=#

#uses implicit contraction 
function SSHOPM(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,W::Array{F,2},
				β::F, max_iter::Int,tol::F;update_user::Int=-1) where {F <:AbstractFloat}

	dimension = minimum((A.n,B.n))


	A_Ti, B_Ti = setup_tame_data(A,B)

    x0 = reshape(W,A.n*B.n)
	x0 ./=norm(x0)
    x_k = copy(x0)

    i = 1
    lambda = Inf

    while true

	    x_k_1 = impTTVnodesym(A.n, B.n, x_k, A_Ti, B_Ti)
		#x_k_1 = implicit_contraction(A,B,x_k)

		#println("norm diff is:",norm(x_k_1_test - x_k_1)/norm(x_k_1_test))
        new_lambda = dot(x_k_1,x_k)

        if β != 0.0
            x_k_1 .+= β * x_k
        end

		x_k_1 ./= norm(x_k_1)
		
		if update_user != -1 && i % update_user == 0
			println("λ: $(new_lambda)")
		end


        if abs(new_lambda - lambda) < tol || i >= max_iter
  			return x_k_1, new_lambda
        else
            x_k = copy(x_k_1)
            lambda = new_lambda
            i += 1
        end

    end

end

function SSHOPM(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor,
				U_0::Array{F,2},V_0::Array{F,2}, β::F, max_iter::Int,tol::F;
				max_rank::Int = minimum((A.n,B.n)),update_user::Int=-1) where {F <:AbstractFloat}

	@assert size(U_0,2) == size(V_0,2)

	dimension = minimum((A.n,B.n))

	normalization_factor = sqrt(tr((V_0'*V_0)*(U_0'*U_0)))
	U_0 ./= sqrt(normalization_factor)
	V_0 ./= sqrt(normalization_factor)

	U_k = copy(U_0)
	V_k = copy(V_0)

	best_U::Array{F,2} = copy(U_k)
	best_V::Array{F,2} = copy(U_k)

	λ_k = Inf

	for i in 1:max_iter

		A_comps, B_comps = get_kron_contract_comps(A,B,U_k,V_k)
		
		λ_k_1 = tr((U_k'*A_comps)*(B_comps'*V_k))

		if β != 0.0
			U_temp = hcat(A_comps, sqrt(β) * U_k)
			V_temp = hcat(B_comps, sqrt(β) * V_k)
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

		normalization_factor = sqrt(tr((V_k_1'*V_k_1)*(U_k_1'*U_k_1)))

		U_k_1 ./= sqrt(normalization_factor)
		V_k_1 ./= sqrt(normalization_factor)

		#Y, Z = get_kron_contract_comps(A,B,U_k_1,V_k_1)

		
		if update_user != -1 && i % update_user == 0
			println("λ_$i: $(λ_k_1) -- rank:$(length(singular_indexes))")
		end

		if abs(λ_k_1 - λ_k) < tol || i == max_iter
			return U_k_1, V_k_1, λ_k_1
		end

		#get the low rank factorization for the next one
		U_k = copy(U_k_1)
		V_k = copy(V_k_1)

		λ_k = λ_k_1

	end

	
end