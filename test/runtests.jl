using Test
using Suppressor
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

A = LambdaTAME.load_ThirdOrderSymTensor("test_tensors/test_tensorA.ssten")
B = LambdaTAME.load_ThirdOrderSymTensor("test_tensors/test_tensorA.ssten")

Unweighted_A  = LambdaTAME.load_UnweightedThirdOrderSymTensor("test_tensors/test_tensorA.ssten")
Unweighted_B  = LambdaTAME.load_UnweightedThirdOrderSymTensor("test_tensors/test_tensorA.ssten")


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
        @inferred LambdaTAME.ΛTAME_param_search_profiled(A,B)
        @inferred LambdaTAME.LowRankTAME_param_search_profiled(A,B)
        @inferred LambdaTAME.TAME_param_search_profiled(A,B)

        @inferred LambdaTAME.ΛTAME_param_search(A,B)
        @inferred LambdaTAME.LowRankTAME_param_search(A,B)
        @inferred LambdaTAME.TAME_param_search(A,B)
    end
end


@testset "Contraction" begin

    d = 5
    tol = 1e-15

    U = rand(A.n,d)
    V = rand(B.n,d)
    X = U*V'

    UTOST_y = LambdaTAME.impTTVnodesym(Unweighted_A,Unweighted_B,reshape(X,A.n*B.n))
    #TOST_y = LambdaTAME.implicit_contraction(A,B,reshape(X,A.n*B.n))   #TODO: known problem with code, unused in experiment
    TOST_U, TOST_V = LambdaTAME.get_kron_contract_comps(A,B,U,V)
    LR_TOST_y = LambdaTAME.reshape(TOST_U*TOST_V',A.n*B.n)


    #ensure all vectors are pairwise equal
    #@test norm(UTOST_y   - TOST_y)/norm(TOST_y) < tol
    #@test norm(LR_TOST_y - TOST_y)/norm(TOST_y) < tol
    @test norm(LR_TOST_y - UTOST_y)/norm(UTOST_y) < tol



end


#TODO: this will be removed once the code is incorporated into the package 
include("../src/mat2cooten.jl")

@testset "Motif Sampling" begin 

    seed!(54321)
    n= 100 
    k = 25
    trials = 1000
    clique_size = 4

    A = LambdaTAME.random_geometric_graph(n,k)
    @inferred Array{Int64,2} tensor_from_graph(A, clique_size, trials)
end