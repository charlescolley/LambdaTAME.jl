@testset "Contraction" begin

    d = 5
    tol = 1e-15

    U = rand(A_TOST.n,d)
    V = rand(B_TOST.n,d)
    X = U*V'

    @testset "Kronecker Product Contraction" begin
        UTOST_y = LambdaTAME.impTTVnodesym(A_UTOST,B_UTOST,reshape(X,A_TOST.n*B_TOST.n))
        #TOST_y = LambdaTAME.implicit_contraction(A_TOST,B_TOST,reshape(X,A_TOST.n*B_TOST.n))   #TODO: known problem with code, unused in experiment
        TOST_U, TOST_V = LambdaTAME.get_kron_contract_comps(A_TOST,B_TOST,U,V)
        LR_TOST_y = LambdaTAME.reshape(TOST_U*TOST_V',A_TOST.n*B_TOST.n)
        UST_U, UST_V = LambdaTAME.get_kron_contract_comps(A_UST,B_UST,U,V)
        
        #ensure all vectors are pairwise equal
        #@test norm(UTOST_y   - TOST_y)/norm(TOST_y) < tol
        #@test norm(LR_TOST_y - TOST_y)/norm(TOST_y) < tol
        @test norm(LR_TOST_y - UTOST_y)/norm(UTOST_y) < tol
        @test norm(TOST_U - UST_U)/norm(TOST_U) < tol
        @test norm(TOST_V - UST_V)/norm(TOST_V) < tol
    end

    
    @testset "Multiple Motif Contraction" begin
        #=
        seed!(54321)
        n= 100 
        k = 25
        trials = 1000
        orders = [3,4,5,6]


        x = ones(n,1)
        y = zeros(n,1)
        A = LambdaTAME.random_geometric_graph(n,k)
        tensors = tensors_from_graph(A,orders,trials)

        for i =1:length(orders)
            LambdaTAME.contraction_divide_out!(tensors[i],x,y)
        end


        =#
    end

    @testset "SSHOPM" begin
        
        
    end

end
