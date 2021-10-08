using LambdaTAME: ΛTAME_param_search_profiled, ΛTAME_param_search,  ΛTAME

@testset "ΛTAME" begin 

    β = 0.0
    α = 1.0 
    tol = 1e-6
    max_iter = 15
    

    @testset "Type Stability" begin

        @suppress_out begin
            # --  Third Order Symmetric Tensors  -- #
            @inferred ΛTAME_param_search_profiled(A_TOST,B_TOST)
            @inferred ΛTAME_param_search(A_TOST,B_TOST)
            @inferred ΛTAME(A_TOST,  B_TOST,β,max_iter,tol,α)

            @inferred ΛTAME_param_search_profiled(A_UST,B_UST)
            @inferred ΛTAME_param_search(A_UST,B_UST)
            @inferred ΛTAME(A_UST,  B_UST,β,max_iter,tol,α)

            # --  Unweighted Symmetric Tensors Multi Motif -- #
            @inferred ΛTAME_param_search(A_UST_MM,B_UST_MM)
            @inferred ΛTAME(A_UST_MM,  B_UST_MM,β,max_iter,tol,α)
        end
    end

    @testset "Types output comparisons" begin
        
        # shared 

    
        #=
        # LowRankTAME + TAME only
        d = 5
        U = rand(A_TOST.n,d)
        V = rand(B_TOST.n,d)
        X = U*V'
        =#
        @suppress_out begin
            #no shifts
            U_TOST,  V_TOST   = ΛTAME(A_TOST,  B_TOST,β,max_iter,tol,α)
            U_UST,   V_UST    = ΛTAME(A_UST,   B_UST ,β,max_iter,tol,α)
            U_UST_MM,V_UST_MM = ΛTAME([A_UST],[B_UST],β,max_iter,tol,α) #multimotif routines

            @test norm(U_TOST - U_UST)/norm(U_UST) < NORM_CHECK_TOL
            @test norm(U_UST_MM - U_UST)/norm(U_UST_MM) < NORM_CHECK_TOL
            
            @test norm(V_TOST - V_UST)/norm(V_UST) < NORM_CHECK_TOL
            @test norm(V_UST_MM - V_UST)/norm(V_UST_MM)< NORM_CHECK_TOL
            
            #shifts
            β = 1.0
            α = .5 
            U_TOST,  V_TOST   = ΛTAME(A_TOST,  B_TOST,β,max_iter,tol,α)
            U_UST,   V_UST    = ΛTAME(A_UST,   B_UST ,β,max_iter,tol,α)
            U_UST_MM,V_UST_MM = ΛTAME([A_UST],[B_UST],β,max_iter,tol,α) #multimotif routines

            @test norm(U_TOST - U_UST)/norm(U_UST) < NORM_CHECK_TOL
            @test norm(U_UST_MM - U_UST)/norm(U_UST_MM) < NORM_CHECK_TOL
            
            @test norm(V_TOST - V_UST)/norm(V_UST) < NORM_CHECK_TOL
            @test norm(V_UST_MM - V_UST)/norm(V_UST_MM)< NORM_CHECK_TOL

        end
    end

end