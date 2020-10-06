LambdaTAME
==========

A julia (v1.5) implementation of the LambdaTAME, LowRankTAME, and TAME heurestics for pairwise graph alignments.


Contents
=======
* LambdaTAME.jl:
   Top file
* Experiments.jl:
    Routines for running synthetic experiments or local tensor files 
* Matchings.jl:
    Routines for finding low rank matches. 
* TAME_Implementations.jl:
    Implementations of the TAME, \Lambda-TAME, and low rank TAME routines.
* PostProcessing.jl:
  Routines for the post-processing portion of the TAME algorithm. **in developement**
  
Dependencies
===========
[LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) and  [SparseArrays](https://docs.julialang.org/en/v1/stdlib/SparseArrays/index.html) for sparse numerical linear algebra routines.  [MatrixNetworks](https://github.com/nassarhuda/MatrixNetworks.jl) for finding triangle motifs in arbitrary graphs and random graph generation ([TGPA](https://github.com/eikmeier/TGPA) source code also used for generating the HyperKron models). 


[NPZ](https://github.com/fhs/NPZ.jl), [Random](https://docs.julialang.org/en/v1/stdlib/Random/), and 
[Distributed](https://docs.julialang.org/en/v1/stdlib/Distributed/) for saving, generating, and running experiments (in parallel) respectively. 


[DataStructures](https://github.com/JuliaCollections/DataStructures.jl) used in postprocessing algorithm for finding swap candidates efficiently. 

[Tests](https://docs.julialang.org/en/v1/stdlib/Test/) and [Suppressor](https://github.com/JuliaIO/Suppressor.jl) for testing. 

Data
----
included files are sparse matrix and sparse tensor (.smat and .ssten files respectively) representations of the PPI networks from the [MultiMAGNA++](https://www3.nd.edu/~cone/multiMAGNA++/) project. Data is included for convenience of recreating our experiments. If utilizing these files, please add appropriate citations to their original project source. 

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
* pairwise_alignment


TODO
-------
* convert to module
* remove dependency from ssten 
* add in Testing
* improve Krylov subspace search reuse
