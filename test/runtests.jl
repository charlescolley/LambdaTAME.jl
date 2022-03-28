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
    # norm checks may be scaled by n*m to account for round off error
    #    please see 2.7.6 of 4th edition 
    #    of Matrix Computations for more. 
function round_off_bound(X::Matrix{T}) where T
    return size(X,1)*size(X,2)*NORM_CHECK_TOL
end
function round_off_bound(x::Vector{T}) where T
    return length(x)*NORM_CHECK_TOL
end


tensor_A_file="test_tensors/test_tensorB.ssten"
tensor_B_file="test_tensors/test_tensorA.ssten"
matrix_A_file="test_tensors/testA.smat"
matrix_B_file="test_tensors/testB.smat"
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

A_UST_Cycle = load_SymTensorUnweighted(tensor_A_file,Cycle(),'\t')
B_UST_Cycle = load_SymTensorUnweighted(tensor_B_file,Cycle(),'\t')


#for multi-motif routines
A_UST_MM = [A_UST]#, A_UST]
B_UST_MM = [B_UST]#,] B_UST]

A_UST_Cycle_MM = [A_UST_Cycle]#, A_UST]
B_UST_Cycle_MM = [B_UST_Cycle]#,] B_UST]


# -- test_noerror solution -- #
# src: https://github.com/JuliaLang/julia/issues/18780#issuecomment-863505347
struct NoException <: Exception end
macro test_nothrow(ex)
    esc(:(@test_throws NoException ($(ex); throw(NoException()))))
end


include("Experiments_test.jl")
include("fileio_test.jl")
#include("Contraction_test.jl")
include("Matchings_test.jl")
include("AlignmentDrivers_test.jl")
#include("PostProcessing_test.jl")
include("TAME_implementation_test.jl")
include("LowRankTAME_implementation_test.jl")
include("LambdaTAME_implementation_test.jl")




