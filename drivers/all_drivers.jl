#  -- updated drivers -- #
using Distributions, JSON, DistributedTensorConstruction, Random, Distributed
@everywhere using LambdaTAME

driver_path = "./"
all_results_path = "results/"
sparse_matrices_path = "../data/sparse_matrices/LVGNA/"
sparse_tensors_path = "../data/sparse_tensors/LVGNA/"

function save_results(file_path,data)
    open(file_path,"w") do f
        JSON.print(f,data)
    end
end

include(driver_path*"max_rank_experiments.jl")        # Figure  5.1 (a) 
include(driver_path*"Increasing_Clique_experiments.jl") # Figures 5.1 (b), 5.2 (b), SM3
include(driver_path*"LVGNA_experiments.jl")           # Figures 5.1 (a) 5.3 (a) & (b), SM5
include(driver_path*"RG_Dup_noise_experiments.jl")    # Figures 5.2 (a), SM 4 (a) & (b) 
include(driver_path*"RG_ER_noise_experiments.jl")     # Figures 5.2 (a), SM 4 (a) & (b)
include(driver_path*"singular_values_experiments.jl") # Figure  A.1


function run_all()
    # All functions will run in testing mode unless 
    # 'testing=false' is passed in.

    #  -- max_rank_experiments.jl functions --  #
    LVGNA_max_rank_experiments() 
    LVGNA_shift_experiment() 
    random_graph_experiments()
        
    #  -- Increasing_Clique_experiments.jl functions --  # 
    TAME_LRT_increasing_cliques()               
    increased_clique_size_duplication_noise_experiments()

    #  -- LVGNA_experiments.jl functions --  # 
    LVGNA_alignments()

    #  -- RG_Dup_noise_experiments.jl functions --  # 
    RandomGraph_duplication_noise_experiments()

    #  -- RG_ER_noise_experiments.jl functions --  # 
    RandomGraph_ER_noise_experiments()

    #  -- singular_values_experiments.jl functions --  # 
    singular_values_experiments()
end

 


