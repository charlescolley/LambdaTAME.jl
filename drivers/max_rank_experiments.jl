
function LVGNA_max_rank_experiments(testing=true)
    path = "../data/sparse_tensors/LVGNA/"
    files = readdir(path)
    if testing 
        files = files[6:7]
        files =  ["worm_PHY1.ssten","worm_Y2H1.ssten"]
        alphas = [.5]
        betas = [1e1]
        result_path = all_results_path*"TAME_iterate_max_rank/test/"
    else
        alphas = [.5,1.0]
        betas = [0.0,1e1,1e2,1e3]
        result_path = all_results_path*"TAME_iterate_max_rank/"
    end

    if !isdir(result_path)
        mkpath(result_path)
    end


    X = distributed_pairwise_alignment(files,path,method=LowRankTAME_M(),iter=15;
                                       alphas,betas,profile=true,no_matching=true)
    filename=filter(x-> !isspace(x),"LVGNA_pairwise_alignments_alpha:$(alphas)_beta:$(betas)_iter:15_noMatching_profiled_results.json")
    save_results(result_path*filename,X)

    #=  # Additional drivers for running TAME 
    X = distributed_pairwise_alignment(files,path,method=TAME_M(),iter=15,
                                       alphas,betas,profile=true,no_matching=true)
                                    profile=true,no_matching=true)
    =#
    #TODO: add in saving routines and filenames 
end

function LVGNA_shift_experiment(testing=true)


    tensor_files = readdir(sparse_tensors_path)

    if testing 
        samples = 5
        tol=1e-8
        max_iter=5
        tensor_files = tensor_files[6:7]    
        result_path = all_results_path*"TAME_iterate_max_rank/test/"
    else

        samples = 3000
        tol=1e-16
        max_iter=30


        result_path = all_results_path*"TAME_iterate_max_rank/"
    end
    alphas = [1.0]
    shift_percentage=1.0

    if !isdir(result_path)
        mkpath(result_path)
    end

    foundMaxEigs = []
    for file in tensor_files
        A = load_ThirdOrderSymTensor(sparse_tensors_path*file)
        push!(foundMaxEigs,LambdaTAME.SSHOPM_sample(A,samples,0.0,max_iter,tol)[2])
    end

    rank_profiles = []
    for i = 1:length(tensor_files)
        A = load_ThirdOrderSymTensor(sparse_tensors_path*tensor_files[i])

        for j = i+1:length(tensor_files)
            B = load_ThirdOrderSymTensor(sparse_tensors_path*tensor_files[j])
            β = foundMaxEigs[i]*foundMaxEigs[j]*shift_percentage

            output = align_tensors_profiled(A,B;method = LowRankTAME_M(),alphas=alphas,betas=[β],iter=max_iter,tol=tol,no_matching=true)
            push!(rank_profiles,(tensor_files[i],tensor_files[j],foundMaxEigs[i],foundMaxEigs[j],output.profile))
        end 
    end

    filename=filter(x-> !isspace(x),"LVGNAMaxEigShiftRanks_alphas:$(alphas)_iter:$(max_iter)_SSHOPMSamples:$(samples)_shiftPercentage:$(shift_percentage)_tol:$(tol)_results.json")
    save_results(result_path*filename,rank_profiles)
end

function random_graph_experiments(testing=true)

    if testing 
        n_sizes = [100]
        alphas = [.5]
        betas = [100.0]
        trials = 2
        result_path = all_results_path*"TAME_iterate_max_rank/test/"
    else
        n_sizes = [100, 500, 1000, 2000, 5000, 10000]
        trials = 50
        alphas = [.5,1.0]
        betas = [0.0,1.0,10.0,100.0]
        result_path = all_results_path*"TAME_iterate_max_rank/"
    end

    p_remove = [.05]  #ER param
    edge_inclusion_p = [.5]
    step_percentage = [.25]

    if !isdir(result_path)
        mkpath(result_path)
    end


    X = distributed_random_trials(trials,ErdosRenyiNoise(),true;method=LowRankTAME_M(),graph=RandomGeometric(),iter=15,
                                    degreedist=LogNormal(log(5),1), p_remove,n_sizes,profile=true,no_matching=true,
                                    postProcessing=noPostProcessing(),tol=1e-12,alphas,betas)
    filename=filter(x-> !isspace(x),"LRTAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:$(alphas)_beta:$(betas)_n:$(n_sizes)_p:$(p_remove)_noiseModel:ER_trials:$(trials)_MaxRankResults.json")
    save_results(result_path*filename,X)

   


    X = distributed_random_trials(trials,DuplicationNoise(),true;method=LowRankTAME_M(),graph=RandomGeometric(),iter=15,
                                    degreedist=LogNormal(log(5),1), edge_inclusion_p, step_percentage,n_sizes,
                                    profile=true,no_matching=true, postProcessing=noPostProcessing(),
                                    tol=1e-12,alphas,betas)
    filename=filter(x-> !isspace(x),"LRTAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:$(alphas)_beta:$(betas)_n:$(n_sizes)_p:$(edge_inclusion_p)_noiseModel:Duplication_sp:$(step_percentage)_trials:$(trials)_MaxRankResults.json")
    save_results(result_path*filename,X)

    #=    # Additional drivers for running TAME 
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
    =#

end