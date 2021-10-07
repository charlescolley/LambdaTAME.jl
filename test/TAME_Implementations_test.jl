
@testset "align_tensors (unprofiled)" begin 


    @inferred align_tensors(A_TOST, B_TOST;method=ΛTAME_M(),matchingMethod = ΛTAME_GramMatching())
    @inferred align_tensors(A_TOST, B_TOST;method=ΛTAME_M(),matchingMethod = ΛTAME_rankOneMatching())
    @inferred align_tensors(A_TOST, B_TOST;method=LowRankTAME_M())
    @inferred align_tensors(A_TOST, B_TOST;method=TAME_M())
    @inferred align_tensors(A_TOST, B_TOST;method=ΛTAME_MultiMotif_M())
    

end

@testset "align_tensors_profiled" begin 

    
    #for () in [(A_TOST, B_TOST), (A_UTOST,B_UTOST),(A_UST, B_UST)]
    @inferred align_tensors_profiled(A_TOST, B_TOST;method=ΛTAME_M(),matchingMethod = ΛTAME_GramMatching())
    @inferred align_tensors_profiled(A_TOST, B_TOST;method=ΛTAME_M(),matchingMethod = ΛTAME_rankOneMatching())
    @inferred align_tensors_profiled(A_TOST, B_TOST;method=LowRankTAME_M())
    @inferred align_tensors_profiled(A_TOST, B_TOST;method=TAME_M())
    #@inferred align_tensors_profiled(A_TOST, B_TOST;method=ΛTAME_MultiMotif_M())
        #not-implemnted right now
    #end

end
