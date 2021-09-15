using Test
using Suppressor
using LambdaTAME


#src: https://github.com/charlescolley/DistributedTensorConstruction.jl.git
using DistributedTensorConstruction


import LinearAlgebra: norm
import Random: seed!
#assuming being run from test/ folder
include("../src/LambdaTAME.jl")


#= Potential Kwargs:
   tol - (Float)
   iterations - (Int)
   profile- (bool)
   alphas - (Array{Float64,1})
   betas - (Array{Float64,1})
   rank - Int
=#

#seed 
seed!(54321)

# Third Order Symmetric Tensor
A_TOST = load_ThirdOrderSymTensor("test_tensors/test_tensorA.ssten")
B_TOST = load_ThirdOrderSymTensor("test_tensors/test_tensorB.ssten")

#TODO: rename to A_TOSTU
# Unweighted Third Order Symmetric Tensors
A_UTOST  = load_UnweightedThirdOrderSymTensor("test_tensors/test_tensorA.ssten")
B_UTOST = load_UnweightedThirdOrderSymTensor("test_tensors/test_tensorB.ssten")

# Unweighted Symmetric Tensors
#     src: DistributedTensorConstruction
A_UST = load_SymTensorUnweighted("test_tensors/test_tensorA.ssten",Clique(),'\t')
B_UST = load_SymTensorUnweighted("test_tensors/test_tensorB.ssten",Clique(),'\t')

#for multi-motif routines
A_UST_MM = [A_UST, A_UST]
B_UST_MM = [B_UST, B_UST]

#include()
#include("Contraction_test.jl")
include("Matchings_test.jl")
#include("TAME_Implementations_test.jl")




