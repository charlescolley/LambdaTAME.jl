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
    #=
    @testset "Shared" begin 

        TOST_triangle_count, TOST_gaped_triangles, _ =  TAME_score(A_TOST,A_TOST,mapping)
        #UTOST_triangle_count, UTOST_gaped_triangles, _ =  LambdaTAME.TAME_score(A_UTOST,A_UTOST,mapping)
        UST_triangle_count, UST_gaped_triangles, _ = TAME_score(A_UST,A_UST,mapping)

        #@test TOST_triangle_count == UTOST_triangle_count
        @test TOST_triangle_count == UST_triangle_count

        @test TOST_gaped_triangles == UST_gaped_triangles
        #@test UTOST_gaped_triangles == UST_gaped_triangles

    end
    =#

    @testset "low level matching" begin 

        # TODO: degree_based_matching()
        @inferred low_rank_matching(U,V)
        @test_nothrow low_rank_matching(U,V)
        @inferred rank_one_matching(u,v)
        @test_nothrow rank_one_matching(u,v)

        @testset "bipartite_matching_primal_dual tests" begin 
            @test_nothrow bipartite_matching_primal_dual(X;primalDualTol=1e-5) #solve to a less strict degree

            val,noute,match1,match2 = @inferred bipartite_matching_primal_dual(X;primalDualTol=1e-5)
            val_adj,noute_adj,match1_adj,match2_adj = bipartite_matching_primal_dual(X';primalDualTol=1e-5)

            @test length(match1) == size(X,1) && length(match2) == size(X,2) 
            @test length(match1_adj) == size(X,2) && length(match2_adj) == size(X,1) #ensure match1 is A_to_B
            @test (val == val_adj) && (noute == noute_adj)  #check symmetry
            @test match1 == match2_adj && match2 == match1_adj
        end
    end
    
    #=
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
    =#
    """
    @testset "LowRankTAME" begin 
    end

    @testset "TAME" begin 
    end
    """
    
    @testset "TAME_score" begin 

        #TODO: update
        @testset "Embedding Based" begin

            _ = @inferred TAME_score(A_TOST,B_TOST,U,V)
            _ = @inferred TAME_score(A_TOST,B_TOST,X)
            _ = @inferred TAME_score(A_TOST,B_TOST,x)
            #_ = @inferred TAME_score(A_UTOST,B_UTOST,X)
            _ = @inferred TAME_score(A_UST,B_UST,X)
            _ = @inferred TAME_score(A_UST_MM,B_UST_MM,X)
        end
        #TODO make mapping 

        @testset "Mapping Based" begin 

            @testset "Higher Level" begin

                for (A,B) in [(A_TOST,B_TOST),(A_UST,B_UST),(A_UST_Cycle,B_UST_Cycle),(A_UST_MM,B_UST_MM),(A_UST_Cycle_MM,B_UST_Cycle_MM)]
                    if typeof(A) <: Vector #multimotif must have array accessed
                        m = A[1].n
                        n = B[1].n
                    else
                        m = A.n
                        n = B.n
                    end
                    A_to_B_array = collect(1:m)
                    B_to_A_array = -ones(Int,n)

                    A_to_B_dict = Dict([(i,i) for i in A_to_B_array]) #using an identity map

                    
                    idx = 1
                    while idx <= n && idx <= m
                        B_to_A_array[idx] = idx
                        idx += 1
                    end
                    B_to_A_dict = Dict([(j,i) for (i,j) in A_to_B_dict]) 
                        #matchings may be incomplete

                    #check 
                    for (A_to_B, B_to_A) in [(A_to_B_dict,B_to_A_dict)]#(A_to_B_array,B_to_A_array)]
                        _,gapedMotifs = TAME_score(A,A,A_to_B)
                        if typeof(A) <: Vector
                            _,A_to_B_matchedMotifs  = TAME_score(A,A,A_to_B)
                            @test all([matchedMotif == size(ten.indices,2) for (matchedMotif,ten) in zip(A_to_B_matchedMotifs,A)])
                        else
                            _,gapedMotifs = TAME_score(A,A,A_to_B)
                            @test gapedMotifs == 0
                        end 

                        if typeof(A) <: Vector
                            A_to_B_score,A_to_B_matchedMotifs = TAME_score(A,B,A_to_B)
                            B_to_A_score,B_to_A_matchedMotifs = TAME_score(B,A,B_to_A)
                            check_motifMatch = all([A_to_B_m == B_to_A_m for (A_to_B_m,B_to_A_m) in zip(A_to_B_matchedMotifs,B_to_A_matchedMotifs)])
                            #check_motifMiss = all([A_to_B_g == B_to_A_g for (A_to_B_g,B_to_A_g) in zip(B_to_A_gapedMotifs,B_to_A_gapedMotifs)])
                            @test A_to_B_score == B_to_A_score
                            @test check_motifMatch
                        else
                            A_to_B_matchedMotifs,A_to_B_gapedMotifs = TAME_score(A,B,A_to_B)
                            B_to_A_matchedMotifs,B_to_A_gapedMotifs = TAME_score(B,A,B_to_A)

                            @test (A_to_B_matchedMotifs == B_to_A_matchedMotifs)
                        end
                    end
                end
            end
            @testset "Low Level" begin




            end
        end

    end 


end


