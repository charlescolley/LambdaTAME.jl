#= TODO:

    align_tensors(graph_A_file::String,graph_B_file::String;
                           ThirdOrderSparse=true,kwargs...)

    get_TAME_ranks(graph_A_file::String,graph_B_file::String)

    distributed_pairwise_alignment(dir::String;kwargs...)

    distributed_pairwise_alignment(files::Array{String,1},dirpath::String;kwargs...)

    distributed_TAME_rank_experiment(files::Array{String,1},dirpath::String;kwargs...)

    self_alignment(dir::String;kwargs...)
=#

"""----------------------------------------------------------------------------
    Test case ensuring that all parameter settings for the top level functions 
  can be called without issue

----------------------------------------------------------------------------""

@testset "align_tensors" begin 

    A_file = "test_tensors/test_tensorA.ssten"
    B_file = "test_tensors/test_tensorB.ssten"

    @testset "LambdaTAME" begin
        X = align_tensors(A_file,B_file;profile=true)
        X = align_tensors(A_file,B_file;profile=false)

    end

end
"""


using Distributed: @everywhere, addprocs, remotecall_eval
addprocs(1)
@everywhere using LambdaTAME

@testset "distributed_drivers" begin
    
    trials = 1
    n_sizes = [50] 
    #    header_file_loc = "../src/LambdaTAME.jl"
    #@everywhere include_string(Main,$(read("../src/LambdaTAME.jl",String)),"../src/LambdaTAME.jl")
    #eval(macroexpand(LambdaTAME,quote @everywhere using LambdaTAME end))
   
    #remotecall_eval(Main,[1],using LambdaTAME)
    #using LambdaTAME
    
    @testset "pairwise alignment" begin 



        #@testset "filenames/path functions" begin #TODO: test this. 
        files = [tensor_A_file, tensor_B_file]
        path = "./"
        @suppress_out begin
            for profile in [true,false]
                for method in [ΛTAME_MultiMotif_M(),ΛTAME_M(),LowRankTAME_M(),TAME_M()]
                
                    if method === ΛTAME_MultiMotif_M()
                        @test_broken throw("distributed_pairwise_alignment for ΛTAME_MultiMotif_M is unimplemented.")
                        #=
                        if profile 
                            @test_broken throw("profiling for ΛTAME_MultiMotif_M is unimplemented.")
                        else
                            for motif in [LambdaTAME.Clique(), LambdaTAME.Cycle()]
                                @test_nothrow  distributed_pairwise_alignment(files,path;method,profile,iter=5, samples=10^4,orders=[3],
                                                                                alphas = [1.0,.5],betas = [0.0,1e1],
                                                                                matchingMethod=ΛTAME_GramMatching(),motif,primalDualTol=1e-5)
                                X = distributed_pairwise_alignment(files,path;method,profile,iter=5, samples=10^4,orders=[3],
                                alphas = [1.0,.5],betas = [0.0,1e1],
                                matchingMethod=ΛTAME_GramMatching(),motif,primalDualTol=1e-5)                                                                                
                            end
                        end
                        =#
                    elseif method === ΛTAME_M()
                        @inferred distributed_pairwise_alignment(files,path;method,iter=5, alphas = [1.0,.5],betas = [0.0,1e1],
                                                            matchingMethod=ΛTAME_GramMatching(),profile,primalDualTol=1e-5)
                        @test_nothrow distributed_pairwise_alignment(files,path;method,iter=5, alphas = [1.0,.5],betas = [0.0,1e1],
                                                            matchingMethod=ΛTAME_GramMatching(),profile,primalDualTol=1e-5)
                    elseif method === LowRankTAME_M() || method === TAME_M()
                        @inferred distributed_pairwise_alignment(files,path;method,iter=5, alphas = [1.0,.5],
                                                        betas = [0.0,1e1],profile,primalDualTol=1e-5)
                        @test_nothrow distributed_pairwise_alignment(files,path;method,iter=5, alphas = [1.0,.5],
                                                        betas = [0.0,1e1],profile,primalDualTol=1e-5)
                    end
                end
            end
        end
        #end 

        #@testset "Tensor Object functions" begin 
        #end 
    end


    @testset "random_graphs" begin
        
        @testset "Duplication Noise" begin 
            @suppress_out begin
                for method in [ΛTAME_MultiMotif_M(),ΛTAME_M(),LowRankTAME_M(),TAME_M()]

                    if method === ΛTAME_MultiMotif_M()
                        for motif in [LambdaTAME.Clique(), LambdaTAME.Cycle()]
                            @inferred distributed_random_trials(trials,DuplicationNoise(),true; method, graph=RandomGeometric(), 
                                                    n_sizes, edge_inclusion_p=[1.0], step_percentage=[.1],samples=10^4,orders=[3],
                                                    matchingMethod=ΛTAME_GramMatching(),motif,primalDualTol=1e-5)

                            @test_nothrow distributed_random_trials(trials,DuplicationNoise(),true; method, graph=RandomGeometric(), 
                                                    n_sizes, edge_inclusion_p=[1.0], step_percentage=[.1],samples=10^4,orders=[3],
                                                    matchingMethod=ΛTAME_GramMatching(),motif,primalDualTol=1e-5)
                        end
                    elseif method === ΛTAME_M()
                        @inferred distributed_random_trials(trials,DuplicationNoise(),true; method, graph=RandomGeometric(), 
                                                n_sizes, edge_inclusion_p=[1.0], step_percentage=[.1],
                                                matchingMethod=ΛTAME_GramMatching(),primalDualTol=1e-5)
                        @test_nothrow distributed_random_trials(trials,DuplicationNoise(),true; method, graph=RandomGeometric(), 
                                                n_sizes, edge_inclusion_p=[1.0], step_percentage=[.1],
                                                matchingMethod=ΛTAME_GramMatching(),primalDualTol=1e-5)
                    elseif method === LowRankTAME_M() || method === TAME_M()
                        @inferred distributed_random_trials(trials,DuplicationNoise(),true; method, graph=RandomGeometric(), 
                                                n_sizes, edge_inclusion_p=[1.0], step_percentage=[.1], primalDualTol=1e-5)
                        @test_nothrow distributed_random_trials(trials,DuplicationNoise(),true; method, graph=RandomGeometric(), 
                                                n_sizes, edge_inclusion_p=[1.0], step_percentage=[.1], primalDualTol=1e-5)
                    end
                end
            end
        end

        @testset "Erdos Renyi Noise" begin
            @suppress_out begin
                for method in [ΛTAME_MultiMotif_M(),ΛTAME_M(),LowRankTAME_M(),TAME_M()]
                
                    if method === ΛTAME_MultiMotif_M
                        for motif in [LambdaTAME.Clique(), LambdaTAME.Cycle()]
                            @inferred distributed_random_trials(trials,ErdosRenyiNoise(),true;method, graph=RandomGeometric(), 
                                                                n_sizes, p_remove = [.01], matchingMethod=ΛTAME_GramMatching(),
                                                                samples=10^4,orders=[3],primalDualTol=1e-5)
                            @test_nothrow distributed_random_trials(trials,ErdosRenyiNoise(),true;method, graph=RandomGeometric(), 
                                                                n_sizes, p_remove = [.01], matchingMethod=ΛTAME_GramMatching(),
                                                                samples=10^4,orders=[3],primalDualTol=1e-5)
                        end
                    elseif method === ΛTAME_M
                        @inferred distributed_random_trials(trials,ErdosRenyiNoise(),true;method, graph=RandomGeometric(), 
                                                n_sizes,p_remove = [.01], matchingMethod=ΛTAME_GramMatching(),
                                                primalDualTol=1e-5)
                        @test_nothrow distributed_random_trials(trials,ErdosRenyiNoise(),true;method, graph=RandomGeometric(), 
                                                n_sizes,p_remove = [.01], matchingMethod=ΛTAME_GramMatching(),
                                                primalDualTol=1e-5)

                    elseif method === LowRankTAME_M
                        @inferred distributed_random_trials(trials,ErdosRenyiNoise(),true; method, graph=RandomGeometric(), 
                                                n_sizes, p_remove = [.01], primalDualTol=1e-5)
                        @test_nothrow distributed_random_trials(trials,ErdosRenyiNoise(),true; method, graph=RandomGeometric(), 
                                                n_sizes, p_remove = [.01], primalDualTol=1e-5)
                    elseif method === LowRankTAME_M || method === TAME_M
                        @inferred distributed_random_trials(trials,ErdosRenyiNoise(),true; method, graph=RandomGeometric(), 
                                                n_sizes, p_remove = [.01],primalDualTol=1e-5)
                        @test_nothrow distributed_random_trials(trials,ErdosRenyiNoise(),true; method, graph=RandomGeometric(), 
                                                n_sizes, p_remove = [.01],primalDualTol=1e-5)
                    end
                end
            end
        end
    end

    #
    #    TODO: Check that the primalDualTol parameter works for ErdosNoiseModel, distributed_pairwise_alignment
    #

end