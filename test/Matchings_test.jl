using LambdaTAME: low_rank_matching, rank_one_matching
using LambdaTAME: search_Krylov_space, TAME_score

@testset "Matching Routines" begin 


    U = rand(A_TOST.n,5)
    V = rand(B_TOST.n,5)
    X = U*V'
    x = reshape(X,A_TOST.n*B_TOST.n)
    u = rand(A_TOST.n)
    v = rand(B_TOST.n)

    mapping = Dict([(i,i) for i in 1:A_TOST.n])
    
    @testset "Shared" begin 

        TOST_triangle_count, TOST_gaped_triangles, _ =  TAME_score(A_TOST,A_TOST,mapping)
        #UTOST_triangle_count, UTOST_gaped_triangles, _ =  LambdaTAME.TAME_score(A_UTOST,A_UTOST,mapping)
        UST_triangle_count, UST_gaped_triangles, _ = TAME_score(A_UST,A_UST,mapping)

        #@test TOST_triangle_count == UTOST_triangle_count
        @test TOST_triangle_count == UST_triangle_count

        @test TOST_gaped_triangles == UST_gaped_triangles
        #@test UTOST_gaped_triangles == UST_gaped_triangles

    end

    @testset "low level matching" begin 

        # TODO: degree_based_matching()
        _ = @inferred low_rank_matching(U,V)
        _ = @inferred rank_one_matching(u,v)
        _ = @inferred bipartite_matching_primal_dual(X)

    end

    @testset "ΛTAME" begin 
        d = 5
        U = rand(A_TOST.n,d)
        V = rand(B_TOST.n,d)

        #for returnScoringTimings in [returnTimings(),noTimings()]
            #@isinferred search_Krylov_space(B_TOST,A_TOST,V,U;returnScoringTimings)
            """
            println( search_Krylov_space(B_TOST,A_TOST,V,U;returnScoringTimings))
            @test (@inferred search_Krylov_space(B_TOST,A_TOST,V,U;returnScoringTimings)) == 1
            @test (@inferred search_Krylov_space(B_UST,A_UST,V,U;returnScoringTimings)) == 1
            @test (@inferred search_Krylov_space(B_UST_MM,A_UST_MM,V,U;returnScoringTimings)) == 1
                #work around for @inInferred macro found 
                #   src: https://discourse.julialang.org/t/more-informative-inferred-especially-for-unit-testing/5481/2
            """

        TOST_best_score, TOST_best_i, TOST_best_j, TOST_best_matching = @inferred search_Krylov_space(A_TOST,B_TOST,U,V;returnScoringTimings=returnTimings())
        UST_best_score, UST_best_i, UST_best_j, UST_best_matching = @inferred search_Krylov_space(A_UST,B_UST,U,V;returnScoringTimings=returnTimings())
        UST_MM_best_score,UST_MM_best_matched_motifs, UST_MM_best_i, UST_MM_best_j, UST_MM_best_matching = @inferred search_Krylov_space(A_UST_MM,B_UST_MM,U,V;returnScoringTimings=returnTimings())

        TOST_best_score, TOST_best_i, TOST_best_j, TOST_best_matching = @inferred search_Krylov_space(A_TOST,B_TOST,U,V;returnScoringTimings=noTimings())
        UST_best_score, UST_best_i, UST_best_j, UST_best_matching = @inferred search_Krylov_space(A_UST,B_UST,U,V;returnScoringTimings=noTimings())
        UST_MM_best_score,UST_MM_best_matched_motifs, UST_MM_best_i, UST_MM_best_j, UST_MM_best_matching = @inferred search_Krylov_space(A_UST_MM,B_UST_MM,U,V;returnScoringTimings=noTimings())


        #ThirdOrderSymTensor to UnweightedSymTensor should be the same
        @test TOST_best_score == UST_best_score
        @test UST_best_i == TOST_best_i 
        @test UST_best_j == TOST_best_j
        @test UST_best_matching == TOST_best_matching
        @test UST_best_matching == UST_MM_best_matching

        @test TOST_best_score == UST_MM_best_matched_motifs[end]#/(2*3)
            # MM version score is num of edges, and input is two third order tensors
            # divide by 2 for the two tensors, and 3 for the edges per triangles
        #@test UST_best_i == UST_MM_best_i 
        #@test UST_best_j == UST_MM_best_j 
        @test UST_best_matching == UST_best_matching
        #end
    end
    """
    @testset "LowRankTAME" begin 
    end

    @testset "TAME" begin 
    end
    """
    
    @testset "TAME_score" begin 


        #BUG: B_TOST is bigger than A_TOST, should flip this around 
        _ = @inferred TAME_score(A_TOST,B_TOST,U,V)
        _ = @inferred TAME_score(A_TOST,B_TOST,X)
        _ = @inferred TAME_score(A_TOST,B_TOST,x)
        #_ = @inferred TAME_score(A_UTOST,B_UTOST,X)
        _ = @inferred TAME_score(A_UST,B_UST,X)
        _ = @inferred TAME_score(A_UST_MM,B_UST_MM,X)
        
        #TODO make mapping 

    end 


end


