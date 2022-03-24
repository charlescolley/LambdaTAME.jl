function singular_values_experiments(testing=true)

    if testing
        #  -- test_params --  # 
        n_sizes = [100]
        trials = 2
        result_path = all_results_path*"/rank1_singular_values/test/"
    else
        n_sizes = [100, 500, 1000, 2000, 5000, 10000]
        trials = 50
        result_path = all_results_path*"rank1_singular_values/"
    end

    if !isdir(result_path)
        mkpath(result_path)
    end


    alphas = [1.0]
    betas = [0.0]

    # -- ER params -- # 
    p_remove = [.05] 
    # -- Duplication params -- # 
    edge_inclusion_p = [.5]
    step_percentage = [.25]


    X = distributed_random_trials(trials,ErdosRenyiNoise(),true;method=TAME_M(),graph=RandomGeometric(),iter=15,
                                    degreedist=LogNormal(log(5),1), p_remove,n_sizes,profile=true,no_matching=true,
                                    postProcessing=noPostProcessing(),tol=1e-12,alphas,betas)
    filename=filter(x-> !isspace(x),"TAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:$(alphas)_beta:$(betas)_n:$(n_sizes)_p:$(p_remove)_noiseModel:ER_trials:$(trials)_MaxRankResults.json")
    save_results(result_path*filename,X)


    X = distributed_random_trials(trials,DuplicationNoise(),true;method=TAME_M(),graph=RandomGeometric(),iter=15,
                                    degreedist=LogNormal(log(5),1), edge_inclusion_p, step_percentage,n_sizes,
                                    profile=true,no_matching=true, postProcessing=noPostProcessing(),
                                    tol=1e-12,alphas,betas)
    filename=filter(x-> !isspace(x),"TAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:$(alphas)_beta:$(betas)_n:$(n_sizes)_p:$(edge_inclusion_p)_noiseModel:Duplication_sp:$(step_percentage)_trials:$(trials)_MaxRankResults.json")
    save_results(result_path*filename,X)

end