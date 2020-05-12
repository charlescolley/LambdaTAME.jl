LambdaTAME
==========

A julia implementation of the LambdaTAME heurestic for pairwise graph alignments.

Dependencies
============
* ssten code

Experiments
-----------
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

Display
-------
* print_latex_table
  - remove
* solve_restricted_othorgonal_procrustes_problem
  - remove
* protein_alignmen
  - remove
  
Machinery
---------
* align_tensors
  - low rank 
  - LambdaTAME
  - filenames (tensors are loaded in)
* TAME
* low_rank_matching
* rank_one_matching
* search_Krylov_space
* TAME_score
  - rank 1
  - multi rank
* produce_ssten_from_triangles
* low_rank_TAME
* kron_contract

TODO
----
* add in Testing
* fix low rank
* write TAME
  - check original C code
* write rank growth tests
* improve Krylov subspace search reuse