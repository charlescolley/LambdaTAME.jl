
import LambdaTAME: write_smat
import MatrixNetworks: readSMAT
import SparseArrays: sprand
@testset "fileio tests" begin 

    A = sprand(20,20,.2)
    #A = max.(A,A') 
    @testset "test smat" begin 

        mktempdir() do folder_path

            file = folder_path*"test_mat" #cleanup=true default should delete file
            @test_throws AssertionError write_smat(A,file;delimiter=" ") #no .smat name

            file *=".smat"
            write_smat(A,file;delimeter=' ')
            B = readSMAT(file)
            @test A == B
        end

    end 


end