LambdaTAME
==========

A julia (v1.3) implementation of the LambdaTAME heurestic for pairwise graph alignments.


Contents
=======
* LambdaTAME.jl:
   Top file
* Experiments.jl:
    Routines for running synthetic experiments or local tensor files 
* Matchings.jl:
    Routines for finding low rank matches. 
* TAME_Implementations.jl:
    Implementations of the TAME, \Lambda-TAME, and low rank TAME routines. **some routines are experimental**
* PostProcessing.jl:
  Routines for the post-processing portion of the TAME algorithm. **in developement**
  
Dependencies
===========
[LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) and  [SparseArrays](https://docs.julialang.org/en/v1/stdlib/SparseArrays/index.html) for sparse numerical linear algebra routines.  [MatrixNetworks](https://github.com/nassarhuda/MatrixNetworks.jl) for finding triangle motifs in arbitrary graphs and random graph generation ([TGPA](https://github.com/eikmeier/TGPA) source code also used for generating the HyperKron models). 


[NPZ](https://github.com/fhs/NPZ.jl), [Random](https://docs.julialang.org/en/v1/stdlib/Random/), and 
[Distributed](https://docs.julialang.org/en/v1/stdlib/Distributed/) for saving, generating, and running experiments (in parallel) respectively. 


[DataStructures](https://github.com/JuliaCollections/DataStructures.jl) used in postprocessing algorithm for finding swap candidates efficiently. 
  
Experiments
--------------
* distributed_random_trials
* full_ER_TAME_test
  - synthetic_HyperKron_problem
* full_HyperKron_TAME_test
  - synthetic_HyperKron_problem
* self_alignment
  - file
  - full directory
* distributed_pairwise_alignment
* synthetic_HyperKron_problem
* pairwise_alignmen


TODO
-------
* convert to module
* remove dependency from ssten 
* add in Testing
* fix low rank
* write TAME
  - check original C code
* write rank growth tests
* improve Krylov subspace search reuse
