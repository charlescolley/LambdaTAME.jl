using LambdaTAME: LowRankTAME_param_search_profiled,  LowRankTAME_param_search
using LambdaTAME: LowRankTAME, LowRankTAME_profiled, TAME

import MatrixNetworks: readSMAT

@testset "LowRankTAME" begin 
    # shared 
    β = 1.0
    α = 1.0 
    tol = 1e-6
    max_iter = 15
    
    d = 1
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
        TOST_U, TOST_V, TOST_triangle_count, _ = LowRankTAME(A_TOST, B_TOST, U,V, β, max_iter, tol, α)
        UST_U,UST_V, UST_triangle_count, _ = LowRankTAME(A_UST, B_UST, U,V, β, max_iter, tol, α)

        @test norm(TOST_V - UST_V)/norm(TOST_V) < round_off_bound(TOST_V)
        @test norm(TOST_U - UST_U)/norm(TOST_U) < round_off_bound(TOST_U)

    end

    @testset "LowRankTAME === TAME" begin

        @testset "TOST based methods" begin
            @suppress_out begin
                LRTAME_U, LRTAME_V, LRTAME_triangle_count, LRTAME_mapping = LowRankTAME(A_TOST, B_TOST, U,V, β, max_iter, tol, α)
                TAME_X, TAME_triangle_count, TAME_mapping = TAME(A_TOST, B_TOST, β, max_iter, tol, α;W=X)
                @test TAME_mapping == LRTAME_mapping && LRTAME_triangle_count == TAME_triangle_count
                @test norm(LRTAME_U*LRTAME_V' - TAME_X)/norm(TAME_X) < round_off_bound(TAME_X)
            end
        end


        @testset "UST based methods" begin
            A = readSMAT(matrix_A_file)
            B = readSMAT(matrix_B_file)
            
            U = rand(size(A,1),d)
            V = rand(size(B,1),d)
            X = U*V'
                #test mats are bigger than A_TOST tensors
            for k = 4:7
                A_ten = tensor_from_graph(A,k,Clique())
                B_ten = tensor_from_graph(B,k,Clique())

                @suppress_out begin
                    LRTAME_U, LRTAME_V, LRTAME_triangle_count, LRTAME_mapping = LowRankTAME(A_ten, B_ten, U,V, β, max_iter, tol, α)
                    TAME_X, TAME_triangle_count, TAME_mapping = TAME(A_ten, B_ten, β, max_iter, tol, α;W=X)

                    @test TAME_mapping == LRTAME_mapping && LRTAME_triangle_count == TAME_triangle_count
                    @test norm(LRTAME_U*LRTAME_V' - TAME_X)/norm(TAME_X) < round_off_bound(TAME_X)
                    
                end 

            end 
        end
    end
end