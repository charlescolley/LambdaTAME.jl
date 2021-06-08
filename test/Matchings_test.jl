

@testset "Matching Routines" begin 

    mapping = Dict([(i,i) for i in 1:A_TOST.n])
    
    @testset "Shared" begin 

        TOST_triangle_count, TOST_gaped_triangles, _ =  LambdaTAME.TAME_score(A_TOST,A_TOST,mapping)
        #UTOST_triangle_count, UTOST_gaped_triangles, _ =  LambdaTAME.TAME_score(A_UTOST,A_UTOST,mapping)
        UST_triangle_count, UST_gaped_triangles, _ = LambdaTAME.TAME_score(A_UST,A_UST,mapping)

        #@test TOST_triangle_count == UTOST_triangle_count
        @test TOST_triangle_count == UST_triangle_count

        @test TOST_gaped_triangles == UST_gaped_triangles
        #@test UTOST_gaped_triangles == UST_gaped_triangles

    end

    @testset "ΛTAME" begin 
        d = 5
        U = rand(A_TOST.n,d)
        V = rand(B_TOST.n,d)

        print
        @inferred LambdaTAME.search_Krylov_space(B_TOST,A_TOST,V,U)
        @inferred LambdaTAME.search_Krylov_space(B_UST,A_UST,V,U)
        @inferred LambdaTAME.search_Krylov_space(B_UST_MM,A_UST_MM,V,U)

        TOST_best_score, TOST_best_i, TOST_best_j, TOST_best_matching = LambdaTAME.search_Krylov_space(B_TOST,A_TOST,V,U)
        UST_best_score, UST_best_i, UST_best_j, UST_best_matching = LambdaTAME.search_Krylov_space(B_UST,A_UST,V,U)
        UST_MM_best_score,UST_MM_best_matched_motifs, UST_MM_best_i, UST_MM_best_j, UST_MM_best_matching = LambdaTAME.search_Krylov_space(B_UST_MM,A_UST_MM,V,U)


        #ThirdOrderSymTensor to UnweightedSymTensor should be the same
        @test TOST_best_score == UST_best_score
        @test UST_best_i == TOST_best_i 
        @test UST_best_j == TOST_best_j
        @test UST_best_matching == TOST_best_matching

        @test TOST_best_score == UST_MM_best_score/(2*3)
            # MM version score is num of edges, and input is two third order tensors
            # divide by 2 for the two tensors, and 3 for the edges per triangles
        #@test UST_best_i == UST_MM_best_i 
        #@test UST_best_j == UST_MM_best_j 
        @test UST_best_matching == UST_best_matching
    end

    @testset "LowRankTAME" begin 
    end

    @testset "TAME" begin 
    end


end


