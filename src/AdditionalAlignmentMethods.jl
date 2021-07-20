

"""
      Simplified version of LowRankEigenAlign to only make use of the aligned 
    edges, rather than considering non-matches, and punishing mis-matches. 
"""
function lowRankEigenAlignEdgesOnly(A::SparseMatrixCSC{T, Int64}, B::SparseMatrixCSC{T, Int64}) where T 

    m = size(A,1)
    n = size(B,1)

    tol=1e-8
    maxIter = 50 #likely to converge to desired tolerance

    v,_ = powerMethod(A, ones(m), tol, maxIter)
    u,_ = powerMethod(B, ones(n), tol, maxIter)
    mapping, _ = rank_one_matching(v,u)

    return mapping 

end

#used for the LowRankEigenAlignEdgesOnly routine
function powerMethod(A::SparseMatrixCSC{T, Int64},x0::Array{T,1},tol::Float64,maxIter::Int;update_user =-1) where T 

    y_k = copy(x0)
    y_k ./=norm(y_k)

    λ_k = Inf
    
    for i =1:maxIter

  
        y_k_1 = A*y_k
        λ_k_1 = y_k_1'*y_k 

        y_k_1 ./=norm(y_k_1)

        if update_user != -1 && i % update_user == 0 
            println("step $i : λ_k = $λ_k_1")
        end

        if abs(λ_k - λ_k_1) < tol || i == maxIter
            return y_k_1, λ_k_1
        else 
            λ_k = λ_k_1
            y_k = y_k_1
        end

    end

end