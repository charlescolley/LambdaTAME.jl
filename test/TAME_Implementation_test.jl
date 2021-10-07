using LambdaTAME: TAME_param_search_profiled,  TAME_param_search


@testset "Type Stability" begin

    @suppress_out begin
        # --  Third Order Symmetric Tensors  -- #
        
        @inferred TAME_param_search_profiled(B_TOST,A_TOST)
        @inferred TAME_param_search(B_TOST,A_TOST)

        # --  Unweighted Symmetric Tensors  -- #

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

    #TODO: finish
end