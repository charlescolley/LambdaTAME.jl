
"""-----------------------------------------------------------------------------
    contracts a tensor represented with indices + values in a symmetric COO
  format. Rows of the indices correspond to sorted indices of a symmetric
  hyper-edge. Values correspond to the edge weights, and v is vector being
  contracted with the
  -----------------------------------------------------------------------------"""
function tensor_vector_contraction(A::ThirdOrderSymTensor, x::Array{Float64,1})
    @assert length(x) == A.n

    y = zeros(Float64,A.n)
    edge_count = size(A.indices,1)
	j =1
    @inbounds for (i_1,i_2,i_3) in eachrow(A.indices)

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

    @inbounds for (i_1,i_2,i_3) in eachrow(A.indices)

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
	nnzs = size(A.indices,1)
	I = zeros(Int64,6*nnzs)
	J = zeros(Int64,6*nnzs)
	V = zeros(Float64,6*nnzs)

    @inbounds for nnz_index = 1:nnzs
		index += 6
		i,j,k = A.indices[nnz_index,:]
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
	nnzs = size(A.indices,1)
    for nnz_index = 1:nnzs
		index += 6
		i,j,k = A.indices[nnz_index,:]
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
	A_nnzs = size(A.indices,1)
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

	A_nnzs = size(A.indices,1)
	A_I = zeros(Int64,6*A_nnzs)
	A_J = zeros(Int64,6*A_nnzs)
	A_V = zeros(Float64,6*A_nnzs)


	B_nnzs = size(B.indices,1)
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
    @inbounds for (i_1,i_2,i_3) in eachrow(A.indices)

		A_val = A.values[i]
		i += 1
		j = 1
#        for (i_1,i_2,i_3) in permutations(A.indices[i,:])
		for (j_1,j_2,j_3) in eachrow(B.indices)


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

"""-----------------------------------------------------------------------------
                         Multi Motif Contraction
-----------------------------------------------------------------------------"""
#TODO: test, correct types
function contract_multi_motif_tensor(simplicial_complexes::Array{Array{Int,2},1},x, y)
    #assuming simplicial_complexes are sorted by their motif order
    for i=1:length(simplicial_complexes)
        contraction_divide_out!(simplicial_complexes[i],x,y)
    end
end

#TODO: test, correct types
function contract_embedded_multi_motif_tensor(simplicial_complexes::Array{Array{Int,2},1},x, y)
    #assuming simplicial_complexes are sorted by their motif order

    max_order = size(simplicial_complexes[end],1)
    println(max_order)
    for i=1:length(simplicial_complexes)
        if i == length(simplicial_complexes)
            contraction_divide_out!(simplicial_complexes[i],x,y)
        else
            embedded_contraction!(simplicial_complexes[i],x,y,max_order)
        end
    end
end

function contraction!(indices::Array{Int,2}, x::Array{Float64,1},y::Array{Float64,1})

    order, edges = size(indices)

    scaling_factor = factorial(order-1)

    for i=1:edges
        for j=1:order

            val = scaling_factor

            for k=1:j 
                val *= x[indices[j,i]]
            end

            for k=j+1:order
                val *= x[indices[j,i]]
            end
            y[indices[j,i]] += val
        end
    end
end


#TODO: test, correct types
function contraction_divide_out!(indices::Array{Int,2}, x::Array{Float64,1},y::Array{Float64,1})

    order, edges = size(indices)
    scaling_factor = factorial(order-1)

    for i=1:edges
        val = scaling_factor
        for j=1:order
            val *= x[indices[j,i]]
        end

        for j=1:order
            y[indices[j,i]]+= val/x[indices[j,i]]
        end
    end
end

#TODO: correct types
function embedded_contraction!(indices::Array{Int,2}, x::Array{T,1},y::Array{T,1},embedded_mode::Int) where T

    order,edges = size(indices)
    #@assert order == 3

    # compute contraction template
    partition = [ x for x in integer_partitions(embedded_mode - order) if length(x) <= order]
    #println(partition)
    partition = [length(y) < order ? vcat(y,zeros(Int,order - length(y))) : y  for y in partition]
                     #pad the entries to all have the same length
    partition = [x .+= 1 for x in partition]
    
    multiplicities::Array{Array{Int,1},1} = unique(hcat([collect(permutations(y)) for y in partition]...))

    scaling_factors = Array{Int,2}(undef,order,length(multiplicities))
    for i =1:length(multiplicities)
        for j=1:order
            row = copy(multiplicities[i])
            row[j] -= 1
            scaling_factors[j,i] = multinomial(row...)
        end
        #push!(scaling_factors,row_factors)
    end

    for idx =1:edges

        for idx2 =1:length(multiplicities)
            for idx3 =1:order
                val::Float64 = scaling_factors[idx3,idx2]

                for idx4=1:idx3-1
                    val *= x[indices[idx4,idx]]^multiplicities[idx2][idx4]
                end
                val *= x[indices[idx3,idx]]^(multiplicities[idx2][idx3] -1)
                for idx4=idx3+1:order
                    val *= x[indices[idx4,idx]]^multiplicities[idx2][idx4]
                end

                y[indices[idx3,idx]] += val
            end
        end
    
    end
end


"""-----------------------------------------------------------------------------
                         Iterative method primatives
-----------------------------------------------------------------------------"""

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