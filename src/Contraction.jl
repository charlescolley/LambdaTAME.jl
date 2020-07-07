
"""-----------------------------------------------------------------------------
    contracts a tensor represented with indices + values in a symmetric COO
  format. Rows of the indices correspond to sorted indices of a symmetric
  hyper-edge. Values correspond to the edge weights, and v is vector being
  contracted with the
"""
function tensor_vector_contraction(A::ThirdOrderSymTensor, x::Array{Float64})
    @assert length(x) == A.n

    y = zeros(Float64,A.n)
    edge_count = size(A.indices,1)

    for j=1:edge_count
        i_1,i_2,i_3 = A.indices[j,:]

        y[i_1] = 2 * x[i_2]*x[i_3]*A.values[j]
        y[i_2] = 2 * x[i_1]*x[i_3]*A.values[j]
        y[i_3] = 2 * x[i_1]*x[i_2]*A.values[j]

    end

    return y
end