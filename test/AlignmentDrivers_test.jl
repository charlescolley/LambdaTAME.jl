
import LambdaTAME: spatial_network

@testset "Alignment Drivers" begin

    A = spatial_network(20,2)
    B = spatial_network(25,2)  #different sizes is good for catching errors

    
    betas=[0.0,1.0]

    @suppress_out begin
        for postProcessing in [KlauAlgo(),SuccessiveKlauAlgo(),LocalSearch(verbose=false)]

            #=
            # -- TAME method tests -- #
            for method in [ΛTAME_M(),LowRankTAME_M()]
                for profile in [true,false]
                    if method === ΛTAME_M()
                    for matchingMethod in [ΛTAME_GramMatching(),ΛTAME_rankOneMatching()]
                        #@inferred align_matrices(A,B,KlauAlgo();method,matchingMethod,betas,profile)
                        #type stability if broken by profile Bool. intending to fix this by making it a type flag.
                        #@test_nothrow 
                        @test_nothrow align_matrices(A,B;postProcessing,method,matchingMethod,betas,profile)
                    end
                    elseif method === LowRankTAME_M()
                        @test_nothrow align_matrices(A,B;postProcessing,method,betas,profile)
                    end
                end
            end
            =#

            # -- LowRankEigenAlign method tests -- #
            @test_nothrow align_matrices(A,B;postProcessing,method=LowRankEigenAlign_M())

        end
    end
end