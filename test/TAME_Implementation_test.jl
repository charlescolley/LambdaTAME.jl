using LambdaTAME: TAME_param_search_profiled,  TAME_param_search, TAME, TAME_profiled

@testset "Type Stability" begin

    @suppress_out begin

        @inferred TAME_param_search_profiled(A_TOST,B_TOST)
        @inferred TAME_param_search(A_TOST,B_TOST)


        tol = 1e-15
        β = 0.0
        α = 1.0 
        tol = 1e-6
        max_iter = 15
        @inferred TAME(A_TOST, B_TOST, β, max_iter,tol,α)
        @inferred TAME_profiled(A_TOST, B_TOST, β, max_iter,tol,α)

    end

end
