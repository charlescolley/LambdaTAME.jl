function RandomGraph_duplication_noise_experiments(testing=true)

    if testing 
        default_n = [20]
        default_p = [0.5]
        default_sp = [.25]
    
        alphas = [.5]
        betas = [1.0]

        trial_count = 1
        parameter_sets = [
            ([50],default_p,default_sp),#size exp
            (default_n,[.9],default_sp),
            (default_n,default_p,[.25]),
        ]

        result_path = all_results_path*"/RG_DupNoise/test/"
    else
        #  --  Parameter Sets  --  #
        default_n = [250]
        default_p = [0.5]
        default_sp = [.25]

        seeded=true
        trial_count = 20
        parameter_sets = [
            (default_n,collect(0.0:.1:1.0),default_sp),
            (default_n,default_p,[0.05,0.1,.2,.25,.3,.4,.5]),
            ([100,250,500,1000,1250,1500],default_p,default_sp)#,#size exp
        ]
        alphas = [.5,1.0]
        betas = [0.0,1.0,1e1,1e2]
        result_path = all_results_path*"/RG_DupNoise/"
    end
    seeded=true
    if !isdir(result_path)
        mkpath(result_path)
    end



    test_set = 0 
    for (n_sizes,edge_inclusion_p,step_percentage) in parameter_sets

        #Lambda TAME results 
        #    Gram Matching KlauAlgo Post Processing
        X = distributed_random_trials(trial_count,DuplicationNoise(),seeded;postProcessing=KlauAlgo(rtype=1,maxiter=500),method=ΛTAME_M(),
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1),matchingMethod=ΛTAME_GramMatching(),
                                    n_sizes, edge_inclusion_p,step_percentage,profile=true,alphas,betas)
        filename=filter(x-> !isspace(x),"LambdaTAME_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:Duplication_p:$(edge_inclusion_p)_sp:$(step_percentage)_postProcess:KlauAlgo_trialcount:$(trial_count).json") #Bug: trialcount shortened because filename is too long
        save_results(result_path*filename,X)
        #    Gram Matching LocalSearch Post Processing    
        X = distributed_random_trials(trial_count,DuplicationNoise(),seeded;postProcessing=LocalSearch(),method=ΛTAME_M(),
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1),matchingMethod=ΛTAME_GramMatching(),
                                    n_sizes, edge_inclusion_p,step_percentage,profile=true,alphas,betas)
        filename=filter(x-> !isspace(x),"LambdaTAME_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:Duplication_p:$(edge_inclusion_p)_sp:$(step_percentage)_postProcess:LocalSearch_trialcount:$(trial_count).json") #Bug: trialcount shortened because filename is too long
        save_results(result_path*filename,X)
        
        #=#   LambdaTAME - ROM  - KlauAlgo Post Processing
        X = distributed_random_trials(trial_count,DuplicationNoise(),seeded;postProcessing=KlauAlgo(),method=ΛTAME_M(),alphas,betas,
                                            graph=RandomGeometric(),degreedist=LogNormal(log(5),1),matchingMethod=ΛTAME_rankOneMatching(),
                                            n_sizes, edge_inclusion_p,step_percentage)
        filename=filter(x-> !isspace(x),"LambdaTAME_ROM_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:Duplication_p:$(edge_inclusion_p)_sp:$(step_percentage)_postProcess:KlauAlgo_trialcount:$(trial_count).json") #Bug: trialcount shortened because filename is too long
        save_results(result_path*filename,X)
        =#

        #Low Rank TAME results 
        #    Local Post Processing
        X = distributed_random_trials(trial_count,DuplicationNoise(),seeded;postProcessing=LocalSearch(),method=LowRankTAME_M(),
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1), n_sizes, edge_inclusion_p,step_percentage,profile=true,alphas,betas)
        filename=filter(x-> !isspace(x),"LowRankTAME_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:Duplication_p:$(edge_inclusion_p)_sp:$(step_percentage)_postProcess:LocalSearch_trialcount:$(trial_count).json") #Bug: trialcount shortened because filename is too long
        save_results(result_path*filename,X)
        #    KlauAlgo Post Processing
        X = distributed_random_trials(trial_count,DuplicationNoise(),seeded;postProcessing=KlauAlgo(),method=LowRankTAME_M(),
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1), n_sizes, edge_inclusion_p,step_percentage,profile=true,alphas,betas)
        filename=filter(x-> !isspace(x),"LowRankTAME_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:Duplication_p:$(edge_inclusion_p)_sp:$(step_percentage)_postProcess:KlauAlgo_trialcount:$(trial_count).json") #Bug: trialcount shortened because filename is too long
        save_results(result_path*filename,X)
        
        #= # low rank matching - KlauAlgo Post Processing
        X = distributed_random_trials(trial_count,DuplicationNoise(),seeded;postProcessing=KlauAlgo(),method=LowRankTAME_M(),
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1), n_sizes, edge_inclusion_p,step_percentage,profile=true,low_rank_matching=true,alphas,betas)
        filename=filter(x-> !isspace(x),"LowRankTAME-lrm_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:Duplication_p:$(edge_inclusion_p)_sp:$(step_percentage)_postProcess:KlauAlgo_trialcount:$(trial_count).json") #Bug: trialcount shortened because filename is too long
        save_results(result_path*filename,X)
        #    low rank matching - LocalSearch Post Processing
        X = distributed_random_trials(trial_count,DuplicationNoise(),seeded;postProcessing=LocalSearch(),method=LowRankTAME_M(),
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1), n_sizes, edge_inclusion_p,step_percentage,profile=true,low_rank_matching=true,alphas,betas)
        filename=filter(x-> !isspace(x),"LowRankTAME-lrm_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:Duplication_p:$(edge_inclusion_p)_sp:$(step_percentage)_postProcess:LocalSearch_trialcount:$(trial_count).json") #Bug: trialcount shortened because filename is too long
        save_results(result_path*filename,X)
        =#
        
        
        X = distributed_random_trials(trial_count,DuplicationNoise(),seeded;postProcessing=LocalSearch(),method=LowRankEigenAlign_M(),
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1), n_sizes, edge_inclusion_p,step_percentage,profile=true)
        filename=filter(x-> !isspace(x),"LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:Duplication_p:$(edge_inclusion_p)_sp:$(step_percentage)_postProcess:LocalSearch_trialcount:$(trial_count).json") 
        save_results(result_path*filename,X)

        X = distributed_random_trials(trial_count,DuplicationNoise(),seeded;postProcessing=KlauAlgo(),method=LowRankEigenAlign_M(),
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1), n_sizes, edge_inclusion_p,step_percentage,profile=true)
        filename=filter(x-> !isspace(x),"LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:Duplication_p:$(edge_inclusion_p)_sp:$(step_percentage)_postProcess:KlauAlgo_trialcount:$(trial_count).json")
        save_results(result_path*filename,X)
  
        println("finished parameter set $(test_set)")
        test_set += 1
    end
end