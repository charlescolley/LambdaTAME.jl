using SparseArrays, Distributions
import LambdaTAME: spatial_network, duplication_perturbation_noise_model, netalignmr, knearest_sparsification, successive_netalignmr, successive_netalignmr_profiled
@testset "Post Processing" begin

    m=10
    steps=10
    p=.5

    A = spatial_network(m,2;degreedist=LogNormal(log(5),1))
    n = m + steps
    B,_ = duplication_perturbation_noise_model(A,steps, p)

    #perm = shuffle(1:size(B,1))#
    #B = B[perm,perm]
    d = 10
    U = rand(size(A,1),d)
    V = rand(size(B,1),d)
    matching = collect(enumerate(1:size(A,1)))
    L = sparse(1:min(n,m),1:min(n,m),1,n,m)
    L += sprand(size(L)...,.1) # add some fill in

    @testset "Klau Algorithm" begin 
        @suppress_out begin     
            @inferred netalignmr(B,A,V,U,5,KlauAlgo())
            @inferred netalignmr(B,A,V,U,matching,5,KlauAlgo())
            @inferred netalignmr(B,A,V,U,KlauAlgo())
            @inferred netalignmr(B,A,V,U,matching,KlauAlgo())
            @inferred netalignmr(B,A,L,KlauAlgo())
        end
    end


    @testset "Successive Klau Algorithm" begin
        @suppress_out begin 
            @inferred successive_netalignmr(B,A,V,U,matching,SuccessiveKlauAlgo(k=5,iterDelta=10,successive_iter=3))
            @inferred successive_netalignmr_profiled(B,A,V,U,matching,SuccessiveKlauAlgo(k=5,iterDelta=10,successive_iter=3))
        end
    end

    @testset "k nearest sparsification" begin
        @inferred knearest_sparsification(V,U,matching,5)
    end

end

