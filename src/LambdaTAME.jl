module LambdaTAME

using LinearAlgebra
using NPZ
using JSON
using Distributed
using SparseArrays
using Random
using MatrixNetworks
using DataStructures
using NearestNeighbors
using Parameters

using Distributions
using Metis

#  https://github.com/charlescolley/DistributedTensorConstruction.jl.git
using DistributedTensorConstruction
# imports SymTensorUnweighted type, tensors_from_graph, contraction_divide_out! 
import DistributedTensorConstruction: contraction

using NetworkAlign

import Statistics: mean
import Arpack: svds
import Combinatorics: permutations, combinations
import StatsBase: sample

#store path to project repo here
PROJECT_PATH = "."

abstract type returnType end 


# https://github.com/eikmeier/TGPA.git
#TGPA_PATH = "Users/ccolley/Code/TGPA"
#include(TGPA_PATH*"/TGPA_generator.jl")


#
# https://github.com/nassarhuda/NetworkAlignment.jl.git
#NETWORK_ALIGNMENT_PATH = "/Users/charlie/Documents/Code/NetworkAlignment.jl/src/"
#NETWORK_ALIGNMENT_PATH = "/homes/ccolley/Documents/Software/NetworkAlignment.jl/src/" #nilpotent
#include(NETWORK_ALIGNMENT_PATH*"NetworkAlignment.jl")


#LOWRANK_EIGENALIGN_PATH = "/Users/charlie/Documents/Code/lowrank_spectral_v1_julia/all_code/include_all.jl" #local
#LOWRANK_EIGENALIGN_PATH = "/homes/ccolley/Documents/Software/lowrank_spectral_v1_julia/all_code/include_all.jl"#nilpotent
#include(LOWRANK_EIGENALIGN_PATH)
#TODO: fix 'WARNING: Method definition' messages


#TODO: Check to see if this can be run straight out of the box.
#local graph repos
MULTIMAGNA=PROJECT_PATH*"/data/sparse_tensors/"

#no checks are run, will lead to undefined behavior if variables aren't as specified
struct ThirdOrderSymTensor
    n::Int                  #assuming > 0
    indices::Array{Int,2}   #assuming no repeated permutations, i != k , k != j , i != j
    values::Array{Float64,1}
end

struct UnweightedThirdOrderSymTensor #hyperedge weights are 1.0
    n::Int                  #assuming > 0
    indices::Vector{Vector{Tuple{Int,Int}}}   #assuming no repeated permutations, i != k , k != j , i != j
end

#TODO:
#   Note that to use SparseSymmetricTensors.jl, must update experiment drivers to
#   take a flag that use the local load function


include("Contraction.jl")
include("Matchings.jl")
include("PostProcessing.jl")
include("AlignmentDrivers.jl")
include("RandomGraphs.jl")
include("fileio.jl")
include("Experiments.jl")
#include("TAME_Implementations.jl")
include("LambdaTAME_implementation.jl")
include("LowRankTAME_implementation.jl")
include("TAME_implementation.jl")
#include("Experimental_code.jl")
include("AdditionalAlignmentMethods.jl")

#include("SparseSymmetricTensorCode.jl")




#  --  From Experiments.jl --  #

#all alignment method flag types
export ΛTAME_M, ΛTAME_MultiMotif_M, LowRankTAME_M, LowRankTAME_MultiMotif_M, TAME_M, TAME_MultiMotif_M, EigenAlign_M, LowRankEigenAlign_M, LowRankEigenAlignOnlyEdges_M, Degree_M ,Random_M
export ΛTAME_GramMatching, ΛTAME_rankOneMatching
#random graph flag types
export ErdosRenyi, RandomGeometric, HyperKron
#noise model flag types
export ErdosRenyiNoise, DuplicationNoise


export distributed_pairwise_alignment 
export distributed_random_trials
export align_tensors # this is the file based version
export align_matrices
export random_graph_exp

export load_ThirdOrderSymTensor, load_UnweightedThirdOrderSymTensor
export graph_to_ThirdOrderTensor

export KlauAlgo, noPostProcessing, SuccessiveKlauAlgo, TabuSearch

#  --  From Matchings.jl --  #
#flag types
export returnTimings,noTimings

export bipartite_matching_primal_dual

#  --  From TAME_Implementations.jl --  #
export align_tensors_profiled
export ΛTAME
export LowRankTAME
export LowRankTAME_profiled
export TAME
export TAME_profiled


end #module 