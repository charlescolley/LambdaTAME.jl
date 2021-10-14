
import LambdaTAME: spatial_network

@testset "Post Processing" begin

    A = spatial_network(20,2)
    B = spatial_network(25,2)  #different sizes is good for catching errors

    
    betas=[0.0,1.0]

    @suppress_out begin
        
        for method in [ΛTAME_M(),LowRankTAME_M()]
            for profile in [true,false]
                if method === ΛTAME_M()
                for matchingMethod in [ΛTAME_GramMatching(),ΛTAME_rankOneMatching()]
                    #@inferred align_matrices(A,B,KlauAlgo();method,matchingMethod,betas,profile)
                    #type stability if broken by profile Bool. intending to fix this by making it a type flag.
                    @test_nothrow align_matrices(A,B;postProcessing=KlauAlgo(),method,matchingMethod,betas,profile)
                end
                elseif method === LowRankTAME_M()
                    println("test")
                    @test_nothrow align_matrices(A,B;postProcessing=KlauAlgo(),method,betas,profile)
                end
            end
        end
    end
end