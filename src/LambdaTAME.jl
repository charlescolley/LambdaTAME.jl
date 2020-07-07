using LinearAlgebra
using NPZ
using Distributed
using SparseArrays
using Random
using MatrixNetworks
using DataStructures

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

include("Contraction.jl")
include("Matchings.jl")
include("PostProcessing.jl")
include("TAME_Implementations.jl")
include("Experiments.jl")