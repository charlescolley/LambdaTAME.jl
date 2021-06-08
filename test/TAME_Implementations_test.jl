
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

    @suppress_out begin
        U_TOST,V_TOST = ΛTAME(A_TOST,B_TOST)
        U_UST,V_TOST = ΛTAME(A_TOST,B_TOST)
        


    end
end
