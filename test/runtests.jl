using Test
using Suppressor
using LambdaTAME


#src: https://github.com/charlescolley/DistributedTensorConstruction.jl.git
using DistributedTensorConstruction


import LinearAlgebra: norm
import Random: seed!


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

NORM_CHECK_TOL = 1e-15

tensor_A_file="test_tensors/test_tensorB.ssten"
tensor_B_file="test_tensors/test_tensorA.ssten"
    #NOTE Flipped to address m >= n assertion error in bipartite_matching_primal_dual

# Third Order Symmetric Tensor
A_TOST = load_ThirdOrderSymTensor(tensor_A_file)
B_TOST = load_ThirdOrderSymTensor(tensor_B_file)

#TODO: rename to A_TOSTU
# Unweighted Third Order Symmetric Tensors
A_UTOST  = load_UnweightedThirdOrderSymTensor(tensor_A_file)
B_UTOST = load_UnweightedThirdOrderSymTensor(tensor_B_file)

# Unweighted Symmetric Tensors
#     src: DistributedTensorConstruction
A_UST = load_SymTensorUnweighted(tensor_A_file,Clique(),'\t')
B_UST = load_SymTensorUnweighted(tensor_B_file,Clique(),'\t')

#for multi-motif routines
A_UST_MM = [A_UST]#, A_UST]
B_UST_MM = [B_UST]#,] B_UST]


# -- test_noerror solution -- #
# src: https://github.com/JuliaLang/julia/issues/18780#issuecomment-863505347
struct NoException <: Exception end
macro test_nothrow(ex)
    esc(:(@test_throws NoException ($(ex); throw(NoException()))))
end


#include()
include("Experiments_test.jl")
#include("Contraction_test.jl")
#include("Matchings_test.jl")
#include("TAME_Implementations_test.jl")
include("TAME_implementation_test.jl")
include("LowRankTAME_implementation_test.jl")
include("LambdaTAME_implementation_test.jl")




