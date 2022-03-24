function LVGNA_alignments(testing=true)

    
    files = readdir(sparse_matrices_path)

    if testing
        files = files[6:7]
        files =  ["worm_PHY1.smat","worm_Y2H1.smat"]
        result_path = all_results_path*"LVGNA_alignments/test/"
        alphas = [1.0]
        betas = [0.0]
        samples = 300000
    else
        result_path = all_results_path*"LVGNA_alignments/"
        alphas = [.5,1.0]
        betas = [0.0,1.0,1e1,1e2]
        samples = 3000000
    end

    orders = [3]
        # generates the tensors using TuranShadow as smat 
        # is needed for post processing. Use ssten files for 
        # previously generated triangle adjacency tensors, but 
        # post processing isn't supported. 

        # _MultiMotif_M() flag is used to generate tensors, 
        # replace with _M() for non-generating routines. 

    if !isdir(result_path)
        mkpath(result_path)
    end

    #
    #  --  LowRankTAME  --  #
    #
    X = distributed_pairwise_alignment(files,sparse_matrices_path;postProcessing=KlauAlgo(),method=LowRankTAME_MultiMotif_M(),iter=15,
                                    alphas,betas,profile=true,no_matching=false,motif=Clique(),orders,samples=samples)
    filename =  filter(x-> !isspace(x),"LVGNA_pairwiseAlignment_LowRankTAME_alphas:$(alphas)_betas:$(betas)_iter:15_order:3_postProcessing:KlauAlgo_profile:true_samples:$(samples)_tol:1e-6_results.json")
    save_results(result_path*filename,X)

    X = distributed_pairwise_alignment(files,sparse_matrices_path;postProcessing=LocalSearch(),method=LowRankTAME_MultiMotif_M(),iter=15,
                                    alphas,betas,profile=true,no_matching=false,motif=Clique(),orders,samples=samples)
    filename =  filter(x-> !isspace(x),"LVGNA_pairwiseAlignment_LowRankTAME_alphas:$(alphas)_betas:$(betas)_iter:15_order:$(orders)_postProcessing:LocalSearch_profile:true_tol:1e-6_results.json")
    save_results(result_path*filename,X)

    #=   Low Rank Matching Routines 
    X = distributed_pairwise_alignment(files,sparse_matrices_path;postProcessing=KlauAlgo(),method=LowRankTAME_MultiMotif_M(),iter=15,
                                    alphas,betas,profile=true,no_matching=false,motif=Clique(),orders,samples=samples,low_rank_matching = true)
    filename =  filter(x-> !isspace(x),LVGNA_pairwiseAlignment_LowRankTAME_lrm_alphas:$(alphas)_betas:$(betas)_iter:15_order:3_postProcessing:KlauAlgo_profile:true_samples:$(samples)_tol:1e-6_results.json")
    save_results(result_path*filename,X)
    
    X = distributed_pairwise_alignment(files,sparse_matrices_path;postProcessing=LocalSearch(),method=LowRankTAME_MultiMotif_M(),iter=15,
                                    alphas,betas,profile=true,no_matching=false,motif=Clique(),orders,samples=samples,low_rank_matching = true)
    filename =  filter(x-> !isspace(x),"LVGNA_pairwiseAlignment_LowRankTAME_lrm_alphas:$(alphas)_betas:$(betas)_iter:15_order:3_postProcessing:LocalSearch_profile:true_samples:$(samples)_tol:1e-6_results.json")
    save_results(result_path*filename,X)
    =#

    

    #
    #  --  Lambda - TAME  --  #
    #
    X = distributed_pairwise_alignment(files,sparse_matrices_path;postProcessing=KlauAlgo(),method=ΛTAME_MultiMotif_M(),iter=15,matchingMethod=ΛTAME_GramMatching(),
                                    alphas,betas,profile=true,no_matching=false,orders,samples=samples)
    filename =  filter(x-> !isspace(x),"LVGNA_pairwiseAlignment_LambdaTAME_alphas:$(alphas)_betas:$(betas)_iter:15_matchingMethod:GramMatching_orders:$(orders)_postProcessing:KlauAlgo_profile:true_tol:1e-6_results.json")
    save_results(result_path*filename,X)

    X = distributed_pairwise_alignment(files,sparse_matrices_path;postProcessing=LocalSearch(),method=ΛTAME_MultiMotif_M(),iter=15,matchingMethod=ΛTAME_GramMatching(),
                                    alphas,betas,profile=true,no_matching=false,motif=Clique(),orders,samples=samples)
    filename =  filter(x-> !isspace(x),"LVGNA_pairwiseAlignment_LambdaTAME_alphas:$(alphas)_betas:$(betas)_iter:15_matchingMethod:GramMatching_orders:$(orders)_postProcessing:LocalSearch_profile:true_samples:$(samples)_tol:1e-6_results.json")
    save_results(result_path*filename,X)

    #= Low Rank Matching Routines 
    #02/11/22 TODO: must fix samples and orders being passed in for calling rankOneMatching

    X = distributed_pairwise_alignment(files,sparse_matrices_path;postProcessing=LocalSearch(),method=ΛTAME_M(),iter=15,matchingMethod=ΛTAME_rankOneMatching(),
                                    alphas,betas,profile=true,no_matching=false,motif=Clique(),orders=[3],samples=samples)
    filename =  filter(x-> !isspace(x)"LVGNA_pairwiseAlignment_LambdaTAME_alphas:$(alphas)_betas:$(betas)_iter:15_matchingMethod:rankOneMatching_order:3_postProcessing:LocalSearch_profile:true_samples:$(samples)_tol:1e-6_results.json")
    save_results(result_path*filename,X)
    =#

    #
    #  -- LowRankEigenAlign --  #
    #
    X = distributed_pairwise_alignment(files,sparse_matrices_path;postProcessing=KlauAlgo(),method=LowRankEigenAlign_M(),profile=true)
    filename =  filter(x-> !isspace(x),"LVGNA_pairwiseAlignment_LREA_postProcessing:KlauAlgo_profile:true_results.json")
    save_results(result_path*filename,X)

    X = distributed_pairwise_alignment(files,sparse_matrices_path;postProcessing=LocalSearch(),method=LowRankEigenAlign_M(),profile=true)
    filename =  filter(x-> !isspace(x),"LVGNA_pairwiseAlignment_LREA_postProcessing:LocalSearch_profile:true_results.json")
    save_results(result_path*filename,X)


    #=
    #
    #   (Julia) TAME implementation  
    #
    X = distributed_pairwise_alignment(files,sparse_matrices_path;method=TAME_M(),iter=15,
                                       alphas = [1.0,.5],betas = [0.0,1e1,1e2,1e3],
                                       profile=true,no_matching=false)
    =#

end