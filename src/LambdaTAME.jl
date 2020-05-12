using LinearAlgebra
using NPZ
using Distributed
using SparseArrays
using Random
using MatrixNetworks

PROJECT_PATH = "/Users/ccolley/PycharmProjects/heresWaldo"
SST_PATH = "/Users/ccolley/Documents/Research/SSSTensor/src/SSSTensor.jl" # local path
#SST_PATH = "/homes/ccolley/Documents/Software/SSSTensor/src/SSSTensor.jl" #Nilpotent path
include(SST_PATH)

# https://github.com/eikmeier/TGPA.git
#TGPA_PATH = "Users/ccolley/Code/TGPA"
#include(TGPA_PATH*"/TGPA_generator.jl")

#local graph repos
MULTIMAGNA="/Users/ccolley/PycharmProjects/Graph_Data/MultiMagnaExamples/MultiMagna_TAME"
PROTEIN_TENSORS = PROJECT_PATH*"/data/Protein_structure_data"

include("TAME_Implementations.jl")
include("Matchings.jl")
include("Experiments.jl")