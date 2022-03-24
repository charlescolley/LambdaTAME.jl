function RandomGraph_ER_noise_experiments(testing=true)

    if testing

        default_n = [20]
        default_p = [0.05]

        alphas = [.5]
        betas = [1.0]

        trial_count = 1
        parameter_sets = [
            ([50,60],default_p),
            (default_n,[0.0,0.05]),
        ]
        result_path = all_results_path*"/RG_ERNoise/test/"

    else
        default_n = [250]
        default_p = [0.05]

        alphas = [.5,1.0]
        betas = [0.0,1.0,1e1,1e2]

        trial_count = 20
        parameter_sets = [
            (default_n,[0.01,0.05,.1,0.15,.2,0.25,0.3,0.4]),
            ([100,250,500,1000,1250,1500],default_p),
        ]
        result_path = all_results_path*"/RG_ERNoise/"
    end
    seeded=true

    if !isdir(result_path)
        mkpath(result_path)
    end


    for (n_sizes, p_remove) in parameter_sets
        #
        #    Lambda TAME  
        #   
        X = distributed_random_trials(trial_count,ErdosRenyiNoise(),seeded;postProcessing=KlauAlgo(),method=ΛTAME_M(),alphas,betas,
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1),n_sizes, p_remove,matchingMethod=ΛTAME_GramMatching(),profile=true)

        filename=filter(x-> !isspace(x),"LambdaTAME_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:ErdosReyni_p:$(p_remove)_postProcess:KlauAlgo_trialcount:$(trial_count).json") #Bug: trialcount shortened because filename is too long
        save_results(result_path*filename,X)
        
        X = distributed_random_trials(trial_count,ErdosRenyiNoise(),seeded;postProcessing=LocalSearch(),method=ΛTAME_M(),alphas,betas,
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1),n_sizes, p_remove,matchingMethod=ΛTAME_GramMatching(),profile=true)
        filename=filter(x-> !isspace(x),"LambdaTAME_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:ErdosReyni_p:$(p_remove)_postProcess:LocalSearch_trialcount:$(trial_count).json")
        save_results(result_path*filename,X)

        #
        #    LowRankTAME  
        #   
        X = distributed_random_trials(trial_count,ErdosRenyiNoise(),seeded;postProcessing=LocalSearch(),method=LowRankTAME_M(),alphas,betas,
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1),n_sizes, p_remove,profile=true)
        filename=filter(x-> !isspace(x),"LowRankTAME_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:ErdosReyni_p:$(p_remove)_postProcess:LocalSearch_trialcount:$(trial_count).json")
        save_results(result_path*filename,X)

        X = distributed_random_trials(trial_count,ErdosRenyiNoise(),seeded;postProcessing=KlauAlgo(),method=LowRankTAME_M(),alphas,betas,
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1),n_sizes, p_remove,profile=true)
        filename=filter(x-> !isspace(x),"LowRankTAME_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:ErdosReyni_p:$(p_remove)_postProcess:KlauAlgo_trialcount:$(trial_count).json")
        save_results(result_path*filename,X)
        #= low rank matching routines 
        X = distributed_random_trials(trial_count,ErdosRenyiNoise(),seeded;postProcessing=KlauAlgo(),method=LowRankTAME_M(),alphas,betas,
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1),n_sizes, p_remove,profile=true,low_rank_matching=true)
        filename=filter(x-> !isspace(x),"LowRankTAME-lrm_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:ErdosReyni_p:$(p_remove)_postProcess:KlauAlgo_trialcount:$(trial_count).json")
        save_results(result_path*filename,X)
        =#

        #
        #    LowRankEigenAlign  
        #   
        X = distributed_random_trials(trial_count,ErdosRenyiNoise(),seeded;postProcessing=LocalSearch(),method=LowRankEigenAlign_M(),
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1), n_sizes, p_remove,profile=true)
        filename=filter(x-> !isspace(x),"LowRankEigenAlign_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:ErdosReyni_p:$(p_remove)_postProcess:LocalSearch_trialcount:$(trial_count).json")
        save_results(result_path*filename,X)

        X = distributed_random_trials(trial_count,ErdosRenyiNoise(),seeded;postProcessing=KlauAlgo(),method=LowRankEigenAlign_M(),
                                    graph=RandomGeometric(),degreedist=LogNormal(log(5),1), n_sizes, p_remove,profile=true)
        filename=filter(x-> !isspace(x),"LowRankEigenAlign_graphType:RG_alphas:$(alphas)_betas:$(betas)_degdist:LogNormal-log5_n:$(n_sizes)_noiseModel:ErdosReyni_p:$(p_remove)_postProcess:KlauAlgo_trialcount:$(trial_count).json") 
        save_results(result_path*filename,X)

    end

end