using LambdaTAME: LowRankTAME_param_search_profiled,  LowRankTAME_param_search
using LambdaTAME: LowRankTAME, LowRankTAME_profiled, TAME



@testset "LowRankTAME" begin 
    # shared 
    β = 0.0
    α = 1.0 
    tol = 1e-6
    max_iter = 15
    
    d = 5
    U = rand(A_TOST.n,d)
    V = rand(B_TOST.n,d)
    X = U*V'
    
    @testset "Type Stability" begin
        @suppress_out begin
            # --  Third Order Symmetric Tensors  -- #
            
            for (A_ten,B_ten) in zip([A_TOST,A_UST],[B_TOST,B_UST])

                @inferred LowRankTAME_param_search_profiled(A_ten,B_ten)
                @inferred LowRankTAME_param_search(A_ten,B_ten)

                @inferred LowRankTAME(A_ten, B_ten,U,V, β, max_iter,tol,α)
                @inferred LowRankTAME_profiled(A_ten, B_ten,U,V, β, max_iter,tol,α)

            end
        end
    end
    

    @testset "LowRankTAME_TOST === LowRankTAME_UST" begin
        TOST_V, TOST_U, TOST_triangle_count, _ = LowRankTAME(A_TOST, B_TOST, U,V, β, max_iter, tol, α)
        UST_V,UST_U, UST_triangle_count, _ = LowRankTAME(A_UST, B_UST, U,V, β, max_iter, tol, α)

        @test_broken norm(TOST_V - UST_V)/norm(TOST_V) < NORM_CHECK_TOL
        @test norm(TOST_U - UST_U)/norm(TOST_U) < NORM_CHECK_TOL

    end

    @testset "LowRankTAME === TAME" begin

        @suppress_out begin
            LRTAME_V, LRTAME_U, LRTAME_triangle_count, LRTAME_mapping = LowRankTAME(A_TOST, B_TOST, U,V, β, max_iter, tol, α)
            TAME_X, TAME_triangle_count, TAME_mapping = TAME(A_TOST, B_TOST, β, max_iter, tol, α;W=X)
            @test norm(LRTAME_V*LRTAME_U' - TAME_X)/norm(TAME_X) < NORM_CHECK_TOL
        end

    end
end