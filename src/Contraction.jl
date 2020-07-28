
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

    for j=1:edge_count
        i_1,i_2,i_3 = A.indices[j,:]

        y[i_1] += 2 * x[i_2]*x[i_3]*A.values[j]
        y[i_2] += 2 * x[i_1]*x[i_3]*A.values[j]
        y[i_3] += 2 * x[i_1]*x[i_2]*A.values[j]

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
	return sparse(I[1:index],J[1:index],V[1:index],A.n,A.n)
end

#TODO: adapt to work for any order tensor
function implicit_contraction(A::COOTen,B::COOTen,x::Array{Float64,1})

    @assert length(x) == A.cubical_dimension*B.cubical_dimension
    m = A.cubical_dimension
    n = B.cubical_dimension
    y = zeros(Float64,length(x))

    ileave = (i,j) -> i + m*(j-1)

    for i in 1:length(A)

        for (i_1,i_2,i_3) in permutations(A.indices[i,:])

            for j in 1:length(B)
                j_1,j_2,j_3 = B.indices[j,:]

                y[ileave(i_1,j_1)] += 2*A.vals[i]*B.vals[j]*x[ileave(i_2,j_2)]*x[ileave(i_3,j_3)]
                y[ileave(i_2,j_2)] += 2*A.vals[i]*B.vals[j]*x[ileave(i_1,j_1)]*x[ileave(i_3,j_3)]
                y[ileave(i_3,j_3)] += 2*A.vals[i]*B.vals[j]*x[ileave(i_1,j_1)]*x[ileave(i_2,j_2)]

            end
        end
    end

    return y
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

    result = zeros(A.n * B.n)


    for i in 1:d1

        sub_A_i = tri_sub_tensor(A,U[:,i])
        sub_B_i = tri_sub_tensor(B,V[:,i])
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


function get_kron_contract_comps(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,
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


    for i in 1:d1

        sub_A_i = tri_sub_tensor(A,U[:,i])
        sub_B_i = tri_sub_tensor(B,V[:,i])
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

    @assert length(x) == A.n*B.n
    m = A.n
    n = B.n
    y = zeros(Float64,length(x))

    ileave = (i,j) -> i + m*(j-1)

    for i in 1:size(A.indices,1)

		A_val = A.values[i]

        for (i_1,i_2,i_3) in permutations(A.indices[i,:])

            for j in 1:size(B.indices,1)
                j_1,j_2,j_3 = B.indices[j,:]

                y[ileave(i_1,j_1)] += 2*A_val*B.values[j]*x[ileave(i_2,j_2)]*x[ileave(i_3,j_3)]
                y[ileave(i_2,j_2)] += 2*A_val*B.values[j]*x[ileave(i_1,j_1)]*x[ileave(i_3,j_3)]
                y[ileave(i_3,j_3)] += 2*A_val*B.values[j]*x[ileave(i_1,j_1)]*x[ileave(i_2,j_2)]

            end
        end
    end

    return y
end