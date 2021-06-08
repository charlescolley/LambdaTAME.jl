
@testset "Type Stability" begin

    @suppress_out begin
        # --  Third Order Symmetric Tensors  -- #
        @inferred LambdaTAME.ΛTAME_param_search_profiled(A_TOST,B_TOST)
        @inferred LambdaTAME.LowRankTAME_param_search_profiled(B_TOST,A_TOST)
        @inferred LambdaTAME.TAME_param_search_profiled(B_TOST,A_TOST)

        @inferred LambdaTAME.ΛTAME_param_search(A_TOST,B_TOST)
        @inferred LambdaTAME.LowRankTAME_param_search(B_TOST,A_TOST)
        @inferred LambdaTAME.TAME_param_search(B_TOST,A_TOST)

        # --  Unweighted Symmetric Tensors  -- #

        @inferred LambdaTAME.ΛTAME_param_search(A_UST,B_UST)
        @inferred LambdaTAME.ΛTAME_param_search_profiled(A_UST,B_UST)

        # --  Unweighted Symmetric Tensors Multi Motif -- #
        @inferred LambdaTAME.ΛTAME_param_search(A_UST_MM,B_UST_MM)
    end

end

@testset "Embedding Comparisons" begin

    tol = 1e-15
    β = 0.0
    α = 1.0 
    tol = 1e-6
    max_iter = 15

    @suppress_out begin
        #no shifts
        U_TOST,  V_TOST   = LambdaTAME.ΛTAME(A_TOST,  B_TOST,β,max_iter,tol,α)
        U_UST,   V_UST    = LambdaTAME.ΛTAME(A_UST,   B_UST ,β,max_iter,tol,α)
        U_UST_MM,V_UST_MM = LambdaTAME.ΛTAME([A_UST],[B_UST],β,max_iter,tol,α) #multimotif routines

        @test norm(U_TOST - U_UST)/norm(U_UST) < tol
        @test norm(U_UST_MM - U_UST)/norm(U_UST_MM) < tol
        
        @test norm(V_TOST - V_UST)/norm(V_UST) < tol
        @test norm(V_UST_MM - V_UST)/norm(V_UST_MM)< tol
        
        #shifts
        β = 1.0
        α = .5 
        U_TOST,  V_TOST   = LambdaTAME.ΛTAME(A_TOST,  B_TOST,β,max_iter,tol,α)
        U_UST,   V_UST    = LambdaTAME.ΛTAME(A_UST,   B_UST ,β,max_iter,tol,α)
        U_UST_MM,V_UST_MM = LambdaTAME.ΛTAME([A_UST],[B_UST],β,max_iter,tol,α) #multimotif routines

        @test norm(U_TOST - U_UST)/norm(U_UST) < tol
        @test norm(U_UST_MM - U_UST)/norm(U_UST_MM) < tol
        
        @test norm(V_TOST - V_UST)/norm(V_UST) < tol
        @test norm(V_UST_MM - V_UST)/norm(V_UST_MM)< tol


    end
end
