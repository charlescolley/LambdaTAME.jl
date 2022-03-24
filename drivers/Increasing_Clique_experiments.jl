function TAME_LRT_increasing_cliques(testing=true)

    
    if testing 

        n_sizes = [25]
        orders_to_test = [4]
        samples = 10^3
        trials = 2
        betas=[0.0]
        alphas=[.5]
        results_path = all_results_path*"RG_DupNoise/test/"
 
    else

        n_sizes = [100]
        orders_to_test = [3,4,5,6,7]
        samples = 10^4
        trials = 25
        betas=[0.0,1.0]
        alphas=[.5,1.0]
        results_path = all_results_path*"RG_DupNoise/"

    end
    step_percentage = [.2]
    edge_inclusion_p = .5

    if !isdir(results_path)
        mkpath(results_path)
    end


    LambdaTAME_results = []
    for order in orders_to_test
        X = distributed_random_trials(trials,DuplicationNoise(),true;postProcessing=noPostProcessing(),
                                        n_sizes=n_sizes,edge_inclusion_p,step_percentage,
                                        method=ΛTAME_MultiMotif_M(),graph=RandomGeometric(),degreedist=LogNormal(log(5),1),
                                        betas,profile=true,orders=[order],samples=samples)
        push!(LambdaTAME_results,(order,X))
    end
    filename=filter(x-> !isspace(x),"LambdaTAME_RandomGeometric_degreedist:LogNormal_alpha:$(alphas)_beta:$(betas)_n:$(n_sizes)_orders:$(orders_to_test)_p:$(edge_inclusion_p)_sample:$(samples)_sp:$(step_percentage)_trials:$(trials)_results.json")
    save_results(results_path*filename,LambdaTAME_results)
    
    LowRankTAME_results = []
    for order in orders_to_test
        X = distributed_random_trials(trials,DuplicationNoise(),true;postProcessing=noPostProcessing(),
                                      n_sizes=n_sizes,edge_inclusion_p,step_percentage,method=TAME_MultiMotif_M(),
                                      graph=RandomGeometric(),degreedist=LogNormal(log(5),1),betas,profile=true,
                                      orders=[order],samples=samples)
        push!(LowRankTAME_results,(order,X))                     
    end
    filename=filter(x-> !isspace(x),"LowRankTAME_RandomGeometric_degreedist:LogNormal_alpha:$(alphas)_beta:$(betas)_n:$(n_sizes)_orders:$(orders_to_test)_p:$(edge_inclusion_p)_sample:$(samples)_sp:$(step_percentage)_trials:$(trials)_results.json")
    save_results(results_path*filename,LowRankTAME_results)

    
    TAME_results = []
    for order in orders_to_test
        X = distributed_random_trials(trials,DuplicationNoise(),true;postProcessing=noPostProcessing(),
                                      n_sizes=n_sizes,edge_inclusion_p,step_percentage,
                                      method=LowRankTAME_MultiMotif_M(),graph=RandomGeometric(),degreedist=LogNormal(log(5),1),
                                      betas,profile=true,orders=[order],samples=samples)
        push!(TAME_results,(order,X))  
    end
    filename=filter(x-> !isspace(x),"TAME_RandomGeometric_degreedist:LogNormal_alpha:$(alphas)_beta:$(betas)_n:$(n_sizes)_orders:$(orders_to_test)_p:$(edge_inclusion_p)_sample:$(samples)_sp:$(step_percentage)_trials:$(trials)_results.json")
    save_results(results_path*filename,TAME_results)

end

function increased_clique_size_duplication_noise_experiments(testing=true)

    if testing 
        #  --  test params  --  #
        n_sizes = [500]
        step_percentage = [.1]
        edge_inclusion_p = [.5]
        trials = 1
        samples = 1000
        maxiter = 100
        betas = [1.0]
        alphas = [.5]
        orders_to_test = [3]
        k_vals = [5]
        save = true
        result_path = all_results_path*"RG_DupNoise/test/"

    else
        edge_inclusion_p = [.5]
        n_sizes = [500]
        step_percentage = [.25]
        trials = 25
        samples = 1000000
        maxiter = 1000

        orders_to_test = [2,3,4,5,6,7,8,9]
        #orders_to_test = [3,4,5]
        k_vals = [15,30,45,60,75,90]
        betas = [0.0,1.0,10.0,100.0]
        save = true
        result_path = all_results_path*"RG_DupNoise/"
   
    end
    
    if !isdir(result_path)
        mkpath(result_path)
    end

    LambdaTAME_Klau_results = []
    for order in orders_to_test
        for k in k_vals
            
            X = distributed_random_trials(trials,DuplicationNoise(),true;postProcessing=KlauAlgo(k=k,maxiter=maxiter,rtype=1),n_sizes=n_sizes,edge_inclusion_p=edge_inclusion_p,step_percentage=step_percentage,
                                                            method=ΛTAME_MultiMotif_M(),graph=RandomGeometric(),degreedist=LogNormal(log(5),1),motif=Clique(),matchingMethod=ΛTAME_GramMatching(),
                                                            orders=[order],samples=samples,betas,profile=true) 
            push!(LambdaTAME_Klau_results,(order,k,X))
        end
    end
    filename = filter(x-> !isspace(x),"RandomGeometric_degreedist:LogNormal_KlauAlgokvals:$(k_vals)_noiseModel:Duplication_n:$(n_sizes)_KAmiter:$(maxiter)_orders:$(orders_to_test)_p:$(edge_inclusion_p)_samples:$(samples)_sp:$(step_percentage)_trials:$(trials)_data.json")
    save_results(result_path*filename,LambdaTAME_Klau_results)


    LambdaTAME_LocalSearch_results = []
    for order in orders_to_test
        for k in k_vals
            
            X = distributed_random_trials(trials,DuplicationNoise(),true;postProcessing=LocalSearch(k=k),n_sizes=n_sizes,edge_inclusion_p=edge_inclusion_p, step_percentage=step_percentage,
                                                            method=ΛTAME_MultiMotif_M(),graph=RandomGeometric(),degreedist=LogNormal(log(5),1),motif=Clique(),matchingMethod=ΛTAME_GramMatching(),
                                                            orders=[order],samples=samples,betas,profile=true)  
            push!(LambdaTAME_LocalSearch_results,(order,k,X))
        end
    end
    filename = filter(x-> !isspace(x),"RandomGeometric_degreedist:LogNormal_LocalSearchkvals:$(k_vals)_noiseModel:Duplication_n:$(n_sizes)_orders:$(orders_to_test)_p:$(edge_inclusion_p)_samples:$(samples)_sp:$(step_percentage)_trials:$(trials)_data.json")
    save_results(result_path*filename,LambdaTAME_LocalSearch_results)


end