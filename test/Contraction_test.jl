using LambdaTAME: impTTVnodesym, get_kron_contract_comps

@testset "Contraction" begin

    d = 5

    U = rand(A_TOST.n,d)
    V = rand(B_TOST.n,d)
    X = U*V'

    @testset "Kronecker Product Contraction" begin
        UTOST_y = impTTVnodesym(A_UTOST,B_UTOST,reshape(X,A_TOST.n*B_TOST.n))
        #TOST_y = implicit_contraction(A_TOST,B_TOST,reshape(X,A_TOST.n*B_TOST.n))   #TODO: known problem with code, unused in experiment
        TOST_U, TOST_V = get_kron_contract_comps(A_TOST,B_TOST,U,V)
        TOST_X = TOST_U*TOST_V'
        LR_TOST_y = reshape(TOST_X,A_TOST.n*B_TOST.n)
        UST_U, UST_V = get_kron_contract_comps(A_UST,B_UST,U,V)
        UST_X = UST_U*UST_V'
            # UST based embeddings put factors on one
            # embedding rather than using a sqrt

        @test norm(LR_TOST_y - UTOST_y)/norm(UTOST_y) < round_off_bound(UTOST_y)           
        @test norm(TOST_X - UST_X)/norm(UST_X) < round_off_bound(UST_X)

    end

    @testset "TAME contraction methods" begin 
        
        A_Ti,B_Ti = LambdaTAME.setup_tame_data(A_TOST,B_TOST)
        A_Mi,B_Mi = LambdaTAME.setup_tame_data(A_UST,B_UST)

        TOST_Y = impTTVnodesym(A_TOST.n,B_TOST.n,size(A_TOST.indices,1),X,A_Ti,B_Ti)
        UST_Y = impTTVnodesym(A_UST.n,B_UST.n,size(A_UST.indices,1),X,A_Mi,B_Mi)

        @test norm(TOST_Y - UST_Y)/norm(UST_Y) < round_off_bound(UST_Y)
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
