using LambdaTAME: LowRankTAME_param_search_profiled,  LowRankTAME_param_search


@testset "Type Stability" begin

    @suppress_out begin
        # --  Third Order Symmetric Tensors  -- #
        
        @inferred LowRankTAME_param_search_profiled(B_TOST,A_TOST)
        @inferred LowRankTAME_param_search(B_TOST,A_TOST)

    end
end



@testset "Embedding Comparisons" begin

    # shared 
    tol = 1e-15
    β = 0.0
    α = 1.0 
    tol = 1e-6
    max_iter = 15
    
    d = 5
    U = rand(A_TOST.n,d)
    V = rand(B_TOST.n,d)
    X = U*V'

 
    @testset "LowRankTAME" begin
        @suppress_out begin

            U_TOST, V_TOST, triangle_count_TOST, mapping_TOST = LowRankTAME(A_TOST, B_TOST,U,V, β, max_iter,tol,α)

        end
    end
end
