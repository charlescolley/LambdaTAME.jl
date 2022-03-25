LambdaTAME
==========

A julia implementation of the LambdaTAME, LowRankTAME, and TAME heurestics for pairwise graph alignments.


How to Install
==============
```
# in ']' mode
add https://github.com/charlescolley/NetworkAlign.jl
add https://github.com/charlescolley/DistributedTensorConstruction.jl

add https://github.com/charlescolley/LambdaTAME.jl
```

<details>

<summary>Why do I have to install those packages first?</summary>

NetworkAlign.jl & DistributedTensorConstruction.jl are unregistered package dependencies, which must be installed first.

relevant discourse: https://discourse.julialang.org/t/package-manager-resolve-complaining-of-unsatisfiable-requirements-due-to-no-known-versions/23778

</details>

Examples
========
Aligning the smallest LVGNA networks
```julia
using LambdaTAME 

path = "data/sparse_matrices/LVGNA/"
    # assumes this being run from the project folder
files = readdir(path)

alignment_output = align_matrices(path*files[6],path*files[7];method=ΛTAME_M())
                                  #aligned via absolute paths
```

Aligning a synthetic experiment
```julia 
using LambdaTAME

A = LambdaTAME.spatial_network(25, 2;degreedist=LambdaTAME.LogNormal(log(5),1))
B, duplicated_vertices = LambdaTAME.duplication_perturbation_noise_model(A,10, .5)   
perm = LambdaTAME.shuffle(1:size(B,1))
B = B[perm,perm]

betas = [0.0,1.0,10.0]
    # kwargs can be passed down into alignment methods
alignment_output, postprocessing_output = 
            align_matrices(B,A;method=LowRankTAME_M(),postProcessing=LocalSearch(),
                               betas)
                           #put bigger networks first for better cache performance


using DistributedTensorConstruction
# `_MultiMotif_M()` routines sample the network for motifs 
alignment_output, postprocessing_output = 
            align_matrices(B,A;method=ΛTAME_MultiMotif_M(),postProcessing=KlauAlgo(),
                           motif=Clique(), samples=1000,orders=[4])
                           # using `Clique()` uses TuranShadow 
                           #                            Note: order's form is important. 
                           #                                  multiple motifs are supported
                           #                                  by experimental routines.
``` 
for more examples, please view the `drivers/` folder for Distributed.jl experiment drivers 
which recreate our results in _Dominant Z-Eigenpairs of Tensor Kronecker Products are Decoupled (with applications to Higher-Order Graph Matching)_.

Contents
=======
* LambdaTAME.jl:
   Top file.
* Experiments.jl:
    Routines for running synthetic experiments or local tensor files.
* RandomGraphs.jl:
    Routines for building random graphs. 
* Matchings.jl:
    Routines for finding low rank matches. 
* [...]_implementation.jl:
    Implementations of the TAME, \Lambda-TAME, and low rank TAME routines.
* Contractions.jl:
    Routines for computing the tensor contraction operations.
* PostProcessing.jl:
  Routines for new k-nearest neighbor augmentations of LocalSearch and Klau's algorithm.
  
Dependencies
===========
[LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) and  [SparseArrays](https://docs.julialang.org/en/v1/stdlib/SparseArrays/index.html) for sparse numerical linear algebra routines.  [MatrixNetworks](https://github.com/nassarhuda/MatrixNetworks.jl) for finding triangle motifs in arbitrary graphs and random graph generation ([TGPA](https://github.com/eikmeier/TGPA) source code also used for generating the HyperKron models). 


[NPZ](https://github.com/fhs/NPZ.jl), [Random](https://docs.julialang.org/en/v1/stdlib/Random/), and 
[Distributed](https://docs.julialang.org/en/v1/stdlib/Distributed/) for saving, generating, and running experiments (in parallel) respectively. 


[DataStructures](https://github.com/JuliaCollections/DataStructures.jl) used in postprocessing algorithm for finding swap candidates efficiently. [Parameters](https://github.com/mauro3/Parameters.jl) is used to insert
algorithm parameters into type flags to pass parameters in a type stable fashion.

[Tests](https://docs.julialang.org/en/v1/stdlib/Test/) and [Suppressor](https://github.com/JuliaIO/Suppressor.jl) for testing. 

[NetworkAlign.jl](https://github.com/charlescolley/NetworkAlign.jl) is a fork of [Huda Nassar's repository](https://github.com/nassarhuda/NetworkAlign.jl) updated to v1 with some generalized type signatures for ease of use. [DistributedTensorConstruction.jl](https://github.com/charlescolley/DistributedTensorConstruction.jl) is an experimental package containing the routines used for building adjacency tensors with sampled motifs. 

Data
====
included files are sparse matrix and sparse tensor (.smat and .ssten files respectively) representations of the subset of PPI networks from the [LVGNA](https://www3.nd.edu/~cone/LNA_GNA/) project which we use (original .gw files can be directly downloaded [here](https://www3.nd.edu/~cone/LNA_GNA/networks.zip)). Data is included for convenience of recreating our experiments. If utilizing these files, please add appropriate citations to their original project source. 



