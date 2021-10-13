using SparseArrays, Distributions
import LambdaTAME: spatial_network, duplication_perturbation_noise_model, netalignmr
@testset "Klau's Algo" begin

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
    @suppress_out begin     
        @inferred netalignmr(B,A,V,U,5)
        @inferred netalignmr(B,A,V,U,matching,5)
        @inferred netalignmr(B,A,V,U)
        @inferred netalignmr(B,A,V,U,matching)
        @inferred netalignmr(B,A,L)
    end
end