using LinearAlgebra
using NPZ
using Distributed
using SparseArrays
using Random
using MatrixNetworks
using DataStructures
using Hungarian

import Combinatorics: permutations


#store path to project repo here
PROJECT_PATH = "."



SST_PATH = "/Users/ccolley/Documents/Research/SparseSymmetricTensors.jl/src/SparseSymmetricTensors.jl" # local path
#SST_PATH = "/homes/ccolley/Documents/Software/SSSTensor/src/SSSTensor.jl" #Nilpotent path
include(SST_PATH)
using Main.SparseSymmetricTensors

# https://github.com/eikmeier/TGPA.git
#TGPA_PATH = "Users/ccolley/Code/TGPA"
#include(TGPA_PATH*"/TGPA_generator.jl")

#TODO: Check to see if this can be run straight out of the box.
#local graph repos
MULTIMAGNA=PROJECT_PATH*"/data/sparse_tensors"

#no checks are run, will lead to undefined behavior if variables aren't as specified
struct ThirdOrderSymTensor
    n::Int                  #assuming > 0
    indices::Array{Int,2}   #assuming no repeated permutations, i != k , k != j , i != j
    values::Array{Float64,1}
end

#TODO:
#   Note that to use SparseSymmetricTensors.jl, must update experiment drivers to
#   take a flag that use the local load function


include("Contraction.jl")
include("Matchings.jl")
include("PostProcessing.jl")
include("TAME_Implementations.jl")
include("Experimental_code.jl")
include("Experiments.jl")