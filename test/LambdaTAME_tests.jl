using Test
using Suppressor

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

A = load_ThirdOrderSymTensor("test_tensors/test_tensorA.ssten")
B  = load_ThirdOrderSymTensor("test_tensors/test_tensorA.ssten")


#= TODO:

    align_tensors(graph_A_file::String,graph_B_file::String;
                           ThirdOrderSparse=true,kwargs...)

    get_TAME_ranks(graph_A_file::String,graph_B_file::String)

    distributed_pairwise_alignment(dir::String;kwargs...)

    distributed_pairwise_alignment(files::Array{String,1},dirpath::String;kwargs...)

    distributed_TAME_rank_experiment(files::Array{String,1},dirpath::String;kwargs...)

    self_alignment(dir::String;kwargs...)
=#

@testset "Type Stability" begin

    @suppress_out begin
        @inferred ΛTAME_param_search_profiled(A,B)
        @inferred LowRankTAME_param_search_profiled(A,B)
        @inferred TAME_param_search_profiled(A,B)

        @inferred ΛTAME_param_search(A,B)
        @inferred LowRankTAME_param_search(A,B)
        @inferred TAME_param_search(A,B)
    end
end