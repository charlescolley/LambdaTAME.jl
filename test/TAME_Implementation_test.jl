using LambdaTAME: TAME_param_search_profiled,  TAME_param_search, TAME, TAME_profiled
using LambdaTAME: setup_tame_data
@testset "TAME" begin 

    β = 1.0
    α = 0.5 
    tol = 1e-6
    max_iter = 15
    @testset "Type Stability" begin

        @suppress_out begin

            for (A_ten,B_ten) in [(A_TOST,B_TOST),(A_UST,A_UST),(A_UST_Cycle,B_UST_Cycle)]
                @inferred TAME_param_search_profiled(A_TOST,B_TOST)
                @inferred TAME_param_search(A_TOST,B_TOST)

              
                @inferred TAME(A_TOST, B_TOST, β, max_iter,tol,α)
                @inferred TAME_profiled(A_TOST, B_TOST, β, max_iter,tol,α)
            end
        end

    end

    @testset "TOST === UST when k = 3" begin 
        #ensure the code gives the same results

        A_Ti,B_Ti = setup_tame_data(A_TOST,B_TOST)
        A_Mi,B_Mi = setup_tame_data(A_UST,B_UST)

        @testset "setup code" begin 
            @test all([Set([Tuple(x) for x in A_Mi[i]]) == Set(A_Ti[i]) for i in 1:A_UST.n])
            @test all([Set([Tuple(x) for x in B_Mi[i]]) == Set(B_Ti[i]) for i in 1:B_UST.n])
        end 

        @testset "power method" begin 

            x_TOST,tri_match, match_TOST  = TAME(A_TOST, B_TOST, β, max_iter,tol,α)
            x_UST,motif_match, match_UST= TAME(A_UST, B_UST, β, max_iter,tol,α)

            @test norm(x_TOST - x_UST)/norm(x_UST) < NORM_CHECK_TOL
            @test match_UST == match_TOST && tri_match == motif_match
            

        end 

    end 


end
