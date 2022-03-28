using Distributed: @everywhere, addprocs, remotecall_eval
addprocs(1)
@everywhere using LambdaTAME

@testset "distributed_drivers" begin
    
    trials = 1
    n_sizes = [50] 
    
    @testset "pairwise alignment" begin 

        betas = [0.0,1e1]
        alphas = [1.0,.5]
        iter=5
        primalDualTol=1e-5
        # -- check tensor alignment routines -- #
        @testset "Align ssten files " begin 
            files = [tensor_A_file, tensor_B_file]
            path = "./"
            
            @suppress_out begin
                for profile in [true,false]
                    for method in [ΛTAME_M(),LowRankTAME_M(),TAME_M()]
                        if method === ΛTAME_M()
                            @inferred distributed_pairwise_alignment(files,path;method,iter=5, alphas, betas,
                                                                matchingMethod=ΛTAME_GramMatching(),profile,primalDualTol)
                            @test_nothrow distributed_pairwise_alignment(files,path;method,iter=5, alphas,betas,
                                                                matchingMethod=ΛTAME_GramMatching(),profile,primalDualTol)
                        elseif method === LowRankTAME_M() || method === TAME_M()
                            @inferred distributed_pairwise_alignment(files,path;method,iter, alphas,
                                                                     betas, profile, primalDualTol)
                            @test_nothrow distributed_pairwise_alignment(files,path;method, iter, alphas,
                                                                         betas,profile,primalDualTol)
                        end
                    end
                end
            end
        end
        
        @testset "Align smat files " begin 
            @suppress_out begin 
                files = [matrix_A_file, matrix_B_file]
                path = "./"

                samples=1000
                orders = [4]
                motif = LambdaTAME.Clique() 
                        # Cycles are partially supported
                for profile in [true,false]
                    for postProcessing in [noPostProcessing(),KlauAlgo(maxiter=2),LocalSearch(iterations=2)]
                        for method in [ΛTAME_MultiMotif_M(),LowRankTAME_MultiMotif_M(),ΛTAME_M(),LowRankTAME_M(),LowRankEigenAlign_M()]
                            if method === ΛTAME_MultiMotif_M()
                        
                                @inferred distributed_pairwise_alignment(files,path;method,profile,alphas,betas,primalDualTol,
                                                                                    motif,samples,orders,postProcessing,
                                                                                    matchingMethod=ΛTAME_GramMatching())
                                @test_nothrow distributed_pairwise_alignment(files,path;method,profile,alphas,betas,primalDualTol,
                                                                                        motif,samples,orders,postProcessing,
                                                                                        matchingMethod=ΛTAME_GramMatching())                                                                              
                            elseif method === LowRankTAME_MultiMotif_M() || method === TAME_MultiMotif_M()
                                @inferred distributed_pairwise_alignment(files,path;method,iter,alphas,betas,profile,primalDualTol,
                                                                                    motif,samples,orders,postProcessing)
                                @test_nothrow distributed_pairwise_alignment(files,path;method,iter,alphas,betas,profile,primalDualTol,
                                                                                        motif,samples,orders,postProcessing)
                            elseif method === ΛTAME_M()
                                @inferred distributed_pairwise_alignment(files,path;method,iter, alphas, betas,postProcessing,
                                                                         matchingMethod=ΛTAME_GramMatching(),profile,primalDualTol)
                                @test_nothrow distributed_pairwise_alignment(files,path;method,iter, alphas, betas,postProcessing,
                                                                             matchingMethod=ΛTAME_GramMatching(),profile,primalDualTol)
                            elseif method === LowRankTAME_M()
                                @inferred distributed_pairwise_alignment(files,path;method,iter=5, alphas,postProcessing,
                                                                         betas,profile,primalDualTol)
                                @test_nothrow distributed_pairwise_alignment(files,path;method,iter, alphas,postProcessing,
                                                                             betas,profile,primalDualTol)
                            elseif method === LowRankEigenAlign_M()
                                @inferred distributed_pairwise_alignment(files,path;method,profile,postProcessing)
                                @test_nothrow distributed_pairwise_alignment(files,path;method,profile,postProcessing)
                            end
                        end
                    end
                    # -- TAME_M() doesn't support Post processing -- # 
                    @inferred distributed_pairwise_alignment(files,path;method=TAME_M(),iter, alphas,
                                                            betas,profile,primalDualTol)
                    @test_nothrow distributed_pairwise_alignment(files,path;method=TAME_M(),iter, alphas,
                                                                 betas,profile,primalDualTol)        
                    @inferred distributed_pairwise_alignment(files,path;method=TAME_MultiMotif_M(),iter,alphas,
                                                             motif,samples,orders,
                                                             betas,profile,primalDualTol)
                    @test_nothrow distributed_pairwise_alignment(files,path;method=TAME_MultiMotif_M(),iter, alphas,
                                                                 motif,samples,orders,
                                                                 betas,profile,primalDualTol)
                end
            end 
        end

    end


    @testset "random_graphs" begin
        samples=10^4
        orders=[3]
        motif=LambdaTAME.Clique()
        primalDualTol=1e-5
        graph = RandomGeometric()
        @testset "Duplication Noise" begin 
            edge_inclusion_p=[1.0]
            step_percentage=[.1]

            @suppress_out begin
                for postProcessing in [noPostProcessing(),KlauAlgo(maxiter=2),LocalSearch(iterations=2)]
                    for method in [ΛTAME_MultiMotif_M(),LowRankTAME_MultiMotif_M(),ΛTAME_M(),LowRankTAME_M(),LowRankEigenAlign_M()]

                        if method === ΛTAME_MultiMotif_M()

                            @inferred distributed_random_trials(trials,DuplicationNoise(),true; method, graph, 
                                                    n_sizes, edge_inclusion_p, step_percentage,samples,orders,
                                                    matchingMethod=ΛTAME_GramMatching(),motif,primalDualTol,postProcessing)

                            @test_nothrow distributed_random_trials(trials,DuplicationNoise(),true; method, graph, 
                                                    n_sizes, edge_inclusion_p, step_percentage,samples,orders,motif,
                                                    matchingMethod=ΛTAME_GramMatching(),primalDualTol,postProcessing)
                        elseif method === LowRankTAME_MultiMotif_M()
                            @inferred distributed_random_trials(trials,DuplicationNoise(),true; method, graph, 
                                                                n_sizes, edge_inclusion_p, step_percentage,samples,orders,motif,
                                                                primalDualTol,postProcessing)

                            @test_nothrow distributed_random_trials(trials,DuplicationNoise(),true; method, graph, 
                                                                    n_sizes, edge_inclusion_p, step_percentage,samples,orders,motif,
                                                                    primalDualTol,postProcessing)
                        elseif method === ΛTAME_M()
                            @inferred distributed_random_trials(trials,DuplicationNoise(),true; method, graph, 
                                                    n_sizes, edge_inclusion_p, step_percentage,
                                                    matchingMethod=ΛTAME_GramMatching(),primalDualTol,postProcessing)
                            @test_nothrow distributed_random_trials(trials,DuplicationNoise(),true; method, graph, 
                                                    n_sizes, edge_inclusion_p, step_percentage,
                                                    matchingMethod=ΛTAME_GramMatching(),primalDualTol,postProcessing)
                        elseif method === LowRankTAME_M()
                            @inferred distributed_random_trials(trials,DuplicationNoise(),true; method, graph, 
                                                    n_sizes, edge_inclusion_p, step_percentage, primalDualTol,postProcessing)
                            @test_nothrow distributed_random_trials(trials,DuplicationNoise(),true; method, graph, 
                                                    n_sizes, edge_inclusion_p, step_percentage, primalDualTol,postProcessing)
                        elseif method === LowRankEigenAlign_M()
                            @inferred distributed_random_trials(trials,DuplicationNoise(),true; method, graph, 
                                                    n_sizes, edge_inclusion_p, step_percentage,postProcessing)
                            @test_nothrow distributed_random_trials(trials,DuplicationNoise(),true; method, graph, 
                                                    n_sizes, edge_inclusion_p, step_percentage,postProcessing)
                        end
                    end
                end

                # -- TAME_M() doesn't support Post processing -- # 
                @inferred distributed_random_trials(trials,DuplicationNoise(),true; method=TAME_M(), graph, 
                                                    n_sizes, edge_inclusion_p, step_percentage, 
                                                    primalDualTol)
                @test_nothrow distributed_random_trials(trials,DuplicationNoise(),true; method=TAME_M(), graph, 
                                                        n_sizes, edge_inclusion_p, step_percentage, 
                                                        primalDualTol)
                @inferred distributed_random_trials(trials,DuplicationNoise(),true; method=TAME_MultiMotif_M(), graph, 
                                                    n_sizes, edge_inclusion_p, step_percentage, 
                                                    samples,orders,motif, primalDualTol)
                @test_nothrow distributed_random_trials(trials,DuplicationNoise(),true; method=TAME_MultiMotif_M(), graph, 
                                                        n_sizes, edge_inclusion_p, step_percentage, 
                                                        samples,orders,motif, primalDualTol)
            end
        end

        @testset "Erdos Renyi Noise" begin
            p_remove = [.01]
            @suppress_out begin
                for postProcessing in [noPostProcessing(),KlauAlgo(maxiter=2),LocalSearch(iterations=2)]
                    for method in [ΛTAME_MultiMotif_M(),LowRankTAME_MultiMotif_M(),ΛTAME_M(),LowRankTAME_M(),LowRankEigenAlign_M()]
                    
                        if method === ΛTAME_MultiMotif_M()
                            @inferred distributed_random_trials(trials,ErdosRenyiNoise(),true;method, graph, 
                                                                n_sizes, p_remove, matchingMethod=ΛTAME_GramMatching(),
                                                                samples,orders,motif,primalDualTol,postProcessing)
                            @test_nothrow distributed_random_trials(trials,ErdosRenyiNoise(),true;method, graph, 
                                                                n_sizes, p_remove, matchingMethod=ΛTAME_GramMatching(),
                                                                samples,orders,motif,primalDualTol,postProcessing)
                        elseif method === LowRankTAME_MultiMotif_M()
                            @inferred distributed_random_trials(trials,ErdosRenyiNoise(),true;method, graph, 
                                                                n_sizes, p_remove, samples,orders,primalDualTol,
                                                                postProcessing, motif)
                            @test_nothrow distributed_random_trials(trials,ErdosRenyiNoise(),true;method, graph, 
                                                                    n_sizes, p_remove, samples,orders,
                                                                    primalDualTol,postProcessing,motif)
                        elseif method === ΛTAME_M()
                            @inferred distributed_random_trials(trials,ErdosRenyiNoise(),true;method, graph, 
                                                                n_sizes,p_remove, matchingMethod=ΛTAME_GramMatching(),
                                                                primalDualTol,postProcessing)
                            @test_nothrow distributed_random_trials(trials,ErdosRenyiNoise(),true;method, graph, 
                                                                    n_sizes,p_remove, matchingMethod=ΛTAME_GramMatching(),
                                                                    primalDualTol,postProcessing)

                        elseif method === LowRankTAME_M()
                            @inferred distributed_random_trials(trials,ErdosRenyiNoise(),true; method, graph, 
                                                                n_sizes, p_remove, primalDualTol,postProcessing)
                            @test_nothrow distributed_random_trials(trials,ErdosRenyiNoise(),true; method, graph, 
                                                                    n_sizes, p_remove, primalDualTol,postProcessing)
                        elseif method === LowRankEigenAlign_M()
                            @inferred distributed_random_trials(trials,ErdosRenyiNoise(),true; method, graph, 
                                                                n_sizes, p_remove,postProcessing)
                            @test_nothrow distributed_random_trials(trials,ErdosRenyiNoise(),true; method, graph, 
                                                                    n_sizes, p_remove,postProcessing)
                        end
                    end
                end

                # -- TAME_M() doesn't support Post processing -- # 
                
                @inferred distributed_random_trials(trials,ErdosRenyiNoise(),true; method=TAME_M(), graph, 
                                                    n_sizes, p_remove,primalDualTol)
                @test_nothrow distributed_random_trials(trials,ErdosRenyiNoise(),true; method=TAME_M(), graph, 
                                                        n_sizes, p_remove,primalDualTol)
                @inferred distributed_random_trials(trials,ErdosRenyiNoise(),true; method=TAME_MultiMotif_M(), graph, 
                                                    n_sizes, p_remove,primalDualTol,samples,orders,motif)
                @test_nothrow distributed_random_trials(trials,ErdosRenyiNoise(),true; method=TAME_MultiMotif_M(), graph, 
                                                        n_sizes, p_remove,primalDualTol,
                                                        samples,orders,motif)
            end
        end
    end


end