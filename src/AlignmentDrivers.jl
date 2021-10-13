abstract type AlignmentMethod end
struct ΛTAME_M <: AlignmentMethod end
struct ΛTAME_MultiMotif_M <: AlignmentMethod end
struct LowRankTAME_M <: AlignmentMethod end
struct TAME_M <: AlignmentMethod end
struct EigenAlign_M <: AlignmentMethod end
struct LowRankEigenAlign_M <: AlignmentMethod end
struct LowRankEigenAlignOnlyEdges_M <: AlignmentMethod end
struct Degree_M <: AlignmentMethod end
struct Random_M <: AlignmentMethod end

"""------------------------------------------------------------------------------
  This function takes in a pair of sparse matrices and aligns them with with the
  specified method. If the method requested is a tensor method, the adjacency 
  tensor is built from the triangles enumerated. Additional kwargs are passed 

  Inputs
  ------
  * A, B - (SparseMatrixCSC{T,Int}):
    The sparse matrices to be aligned. 
  * profile - (Bool):
    Whether or not to profile the methods run. The type of profiling done is 
    dependent on the method run.  
  * method - (String):
    The method to run. Options include 'Degree', 'Random', 'LowRankEigenAlign',
    'LambdaTAME', 'LowRankTAME', "TAME". Currently supporting 'EigenAlign', but 
    may remove in the future as LowRankEigenAlign produces the same result but 
    with much better scalability. 

  Outputs
  -------
  output depends on whether a tensor method is used or not. Matrix based methods 
  return the trianlges in A, triangles in B, triangles matched, the matching 
  used, runtime.
  
  If a tensor method is used, the number of triangles in A and B are returned, in
  addition to whatever is returned by 'align_tensors(_profiled)'.
------------------------------------------------------------------------------"""
function align_matrices(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{S,Int};
                        profile=false,motif=Clique(),kwargs...) where {T,S}

    method = typeof(kwargs[:method])

    A_ten = graph_to_ThirdOrderTensor(A)
    B_ten = graph_to_ThirdOrderTensor(B)
    
    if method === ΛTAME_M || method === LowRankTAME_M  || method === TAME_M
        if profile
            results = align_tensors_profiled(A_ten,B_ten;kwargs...)
        else
            results = align_tensors(A_ten,B_ten;kwargs...)
        end

        return size(A_ten.indices,1), size(B_ten.indices,1), results

    elseif method === ΛTAME_MultiMotif_M

        A_tensors = tensors_from_graph(A,kwargs[:orders],kwargs[:samples],motif)        
        B_tensors = tensors_from_graph(B,kwargs[:orders],kwargs[:samples],motif)
        
        A_motifCounts = [size(x.indices,2) for x in A_tensors]
        B_motifCounts = [size(x.indices,2) for x in B_tensors]

        A_motifDistribution = [contraction(tensor,ones(tensor.n))./factorial(tensor.order-1) for tensor in A_tensors]
        B_motifDistribution = [contraction(tensor,ones(tensor.n))./factorial(tensor.order-1) for tensor in B_tensors]

        #TODO: standardize kwarg consumption
        subkwargs = Dict([(k,v) for (k,v) in kwargs if k != :orders && k != :samples])
        return A_motifCounts, B_motifCounts,A_motifDistribution, B_motifDistribution, align_tensors(A_tensors,B_tensors;subkwargs...)

    elseif method === EigenAlign_M || method === Degree_M || method === Random_M || method === LowRankEigenAlign_M || method === LowRankEigenAlignOnlyEdges_M
        
        if method === LowRankEigenAlign_M
            iters = 10
            (ma,mb,_,_),t = @timed align_networks_eigenalign(A,B,iters,"lowrank_svd_union",3)
            matching = Dict{Int,Int}([i=>j for (i,j) in zip(ma,mb)]) 
        elseif method === LowRankEigenAlignOnlyEdges_M
            matching,t =@timed lowRankEigenAlignEdgesOnly(A,B) 
        elseif method === EigenAlign_M
            (ma,mb),t = @timed NetworkAlignment.EigenAlign(A,B)
            matching = Dict{Int,Int}(zip(ma,mb))
        elseif method === Degree_M
            (ma,mb),t = @timed degree_based_matching(A,B)
            matching = Dict{Int,Int}(zip(ma,mb))
        elseif kwargs[:method] === Random_M
            n,n = size(B)
            
            matching,t = @timed Dict{Int,Int}(enumerate(shuffle(1:n)))
        else
            error("Invalid input, must be ")
        end
        triangle_count, gaped_triangles  = TAME_score(A_ten,B_ten,matching) 
        return size(A_ten.indices,1), size(B_ten.indices,1), triangle_count, matching, t 
    else
        throw(ArgumentError("method must be of type LambdaTAME_M, LowRankTAME_M, TAME_M, EigenAlign_M, LowRankEigenAlign_M, LowRankEigenAlignEdgesOnly_M, Degree_M, or Random_M."))
    end

    
end

"""-----------------------------------------------------------------------------
  This function takes in a sparse matrix and builds a ThirdOrderSymTensor from 
  the triangles present in the networks. If the network isn't symmetric, then 
  it's symmetrized. The bool 'use_lcc' is used to indicate whether or not to find
  the largest strongly connected component before symmetrizing. 
-----------------------------------------------------------------------------"""
function graph_to_ThirdOrderTensor(A;use_lcc=false)

    n,n = size(A)
    
    if use_lcc
        A,_ = largest_component(A)
    end

    if !issymmetric(A)
        println("Symmetrising matrix")
		A = max.(A,A')  #symmetrize for Triangles routine
	end

    #build COOTens from graphs
    tris = collect(MatrixNetworks.triangles(A))
    nnz = length(tris)


    indices = Array{Int,2}(undef,nnz,3)
    for i =1:nnz
        indices[i,1]= tris[i][1]
        indices[i,2]= tris[i][2]
        indices[i,3]= tris[i][3]
    end

    vals = ones(nnz)

    return ThirdOrderSymTensor(n,indices,vals)

end

"""------------------------------------------------------------------------------
  This function aligns graphs using their tensor representations and returns the
  profiled version of the algorithms. These routines call the 'param_search'
  functions for the associated method used.

  Inputs
  ------
  * A, B - (ThirdOrderSymTensor):
	Two third order tensors representing the presence of triangles within the 
	network. A must be larger than B, else the routines will be called with the
	parameters swapped. 
  * method - (String):
	The choice of method used to align the methods. Options include 'LambdaTAME',
	'LowRankTAME', and 'TAME'. 
  * 'no_matching' - (Bool):
	Will not run the matching routines when True. This is useful when studying 
	the ranks of the iterates. Any counts which may rely on the matchings are 
	replaced by -1. 
  Outputs
  -------
  * 'best_TAME_PP_tris' - (Int):
    The largest number of triangles matched over all iterations.
  * 'max_triangle_match' - (Int):
	The maximum number of triangles matchable. This is the minimum between the 
	number of triangles in graphs A and B. 
  * 'best_TAME_PP_x' - (Array{Float,2}):
	The best iterate found over all the alphas and betas specified by the user. 
	when 'LambdaTAME' and 'LowRankTAME' are called, this is replaced by the U
	and V components of the best iterate. 
  * 'best_matching' - (Dict{Int,Int}):
	The best matching found, maps from A to B. When the method flips A and B, 
	the dictionary is also flipped when returned. 
  * profile - (Dict):
	A dictionary storing the profiling results from each of the methods. Please 
	see the '_profiled' versions of the code to see what is returned by each 
	function.
------------------------------------------------------------------------------"""
function align_tensors_profiled(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor;
					            method::AlignmentMethod=ΛTAME_M(),no_matching=false,kwargs...)

	#put larger tensor on the left
    #=
	if B.n > A.n
        println("swapping tensors in 'align_tensors_profiled'. ")
		results =  align_tensors_profiled(B,A;method = method, no_matching=no_matching,kwargs...)
		#flip the matchings if A and B were swapped
		if typeof(method) === ΛTAME_M 
			if kwargs[:matchingMethod] === ΛTAME_rankOneMatching()
				best_TAME_PP_tris, max_triangle_match, U_best, V_best, best_i, best_j, best_matching, profile = results
				return best_TAME_PP_tris, max_triangle_match, V_best, U_best, best_j, best_i, Dict((j,i) for (i,j) in best_matching), profile
			else 
				best_matched_motifs, max_motif_match, U_best, V_best,best_matching, profile = results
				#BUG: best_matching is returning a vector and not a mapping, type stability is broken too
				return best_matched_motifs, max_motif_match, V_best, U_best, Dict((j,i) for (i,j) in enumerate(best_matching))::Dict{Int64,Int64}, profile
			end
		elseif typeof(method) == LowRankTAME_M
			best_TAME_PP_tris, max_triangle_match, U_best, V_best, best_matching,profile = results
			return best_TAME_PP_tris, max_triangle_match, V_best, U_best, Dict((j,i) for (i,j) in best_matching), profile
		elseif typeof(method) === TAME_M
			best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, best_matching,profile = results
            # TODO: must reshape returned best_TAME_PP_x
			return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, Dict((j,i) for (i,j) in best_matching), profile
		end

	end
    =#

	if typeof(method) == ΛTAME_M
		return ΛTAME_param_search_profiled(A,B;kwargs...)
	elseif typeof(method) === LowRankTAME_M
		return LowRankTAME_param_search_profiled(A,B;no_matching = no_matching,kwargs...)
	elseif typeof(method) === TAME_M
		return TAME_param_search_profiled(A,B;no_matching = no_matching,kwargs...)
	else
		throw(ArgumentError("method must be one of 'LambdaTAME', 'LowRankTAME', or 'TAME'."))
	end
end

"""------------------------------------------------------------------------------
  This function aligns graphs using their tensor representations. These routines
  call the param_search functions for the associated method used.


  Inputs
  ------
  * A, B - (ThirdOrderSymTensor,SymTensorUnweighted,Array{SymTensorUnweighted,1}):
	Two third order tensors representing the presence of triangles within the 
	network. A must be larger than B, else the routines will be called with the
	parameters swapped. 
  * method - (AlignmentMethod):
	The choice of method used to align the methods. Options include LambdaTAME_M,
	LowRankTAME_M, and TAME_M. 
  
  Outputs
  -------
	* 'best_TAME_PP_tris' - (Int):
		The largest number of triangles matched over all iterations.
	* 'max_triangle_match' - (Int):
	  The maximum number of triangles matchable. This is the minimum between the 
	  number of triangles in graphs A and B. 
	* 'best_TAME_PP_x' - (Array{Float,2}):
	  The best iterate found over all the alphas and betas specified by the user. 
	  when 'LambdaTAME' and 'LowRankTAME' are called, this is replaced by the U
	  and V components of the best iterate. 

-----------------------------------------------------------------------------"""
function align_tensors(A::Union{ThirdOrderSymTensor,SymTensorUnweighted{S}}, 
	                   B::Union{ThirdOrderSymTensor,SymTensorUnweighted{S}}; 
					   method::AlignmentMethod=ΛTAME_M(),no_matching=false,kwargs...) where {S <: Motif}

	#TODO: test if ternary is a problem
	#put larger tensor on the left
    #=
	if B.n > A.n
		#TODO: this typeof(A) may fail
		results = align_tensors(B,A;method = method, no_matching=no_matching,kwargs...)
		#flip the matchings if A and B were swapped
		if typeof(method) === ΛTAME_M
			if kwargs[:matchingMethod] === ΛTAME_rankOneMatching()

				best_TAME_PP_tris, max_triangle_match, U_best, V_best,best_i, best_j, best_matching = results
				return best_TAME_PP_tris, max_triangle_match, V_best, U_best, best_j, best_i, Dict((j,i) for (i,j) in best_matching)
			else 
				best_matched_motifs, max_motif_match, U_best, V_best, best_matching = results
				return best_matched_motifs, max_motif_match, V_best, U_best, Dict((j,i) for (i,j) in best_matching)
			end
		elseif typeof(method) === LowRankTAME_M
			best_TAME_PP_tris, max_triangle_match, U_best, V_best, best_matching = results
			return best_TAME_PP_tris, max_triangle_match, V_best, U_best, Dict((j,i) for (i,j) in best_matching)
		elseif typeof(method) === TAME_M
			best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, best_matching = results
			return best_TAME_PP_tris, max_triangle_match, best_TAME_PP_x, Dict((j,i) for (i,j) in best_matching)
		end
	end
    =#

	if typeof(method) === ΛTAME_M || typeof(method) === ΛTAME_MultiMotif_M
		return ΛTAME_param_search(A,B;kwargs...)
	elseif typeof(method) === LowRankTAME_M
		return LowRankTAME_param_search(A,B;no_matching = no_matching,kwargs...)
	elseif typeof(method) === TAME_M
		return TAME_param_search(A,B;no_matching = no_matching,kwargs...)
	else
		throw(ArgumentError("method must be one of LambdaTAME_M,ΛTAME_MultiMotif_M, LowRankTAME_M, or TAME_M."))
	end

end

function align_tensors(A::Array{SymTensorUnweighted{S},1}, B::Array{SymTensorUnweighted{S},1}; 
			           method::AlignmentMethod=ΛTAME_M(),no_matching=false,kwargs...) where {S <: Motif}

    #=
	if (B[1].n > A[1].n)
		results = align_tensors(B,A;method = method, no_matching=no_matching,kwargs...)
		return results[1:end-1]..., Dict((j,i) for (i,j) in results[end])
	end
    =#

	if typeof(method) === ΛTAME_M || typeof(method) === ΛTAME_MultiMotif_M
		return ΛTAME_param_search(A,B;kwargs...)
	elseif typeof(method) === LowRankTAME_M
		return LowRankTAME_param_search(A,B;no_matching = no_matching,kwargs...)
	elseif typeof(method) === TAME_M
		return TAME_param_search(A,B;no_matching = no_matching,kwargs...)
	else
		throw(ArgumentError("method must be one of LambdaTAME_M,ΛTAME_MultiMotif_M, LowRankTAME_M, or TAME_M."))
	end

end


function align_tensors(graph_A_file::String,graph_B_file::String;
                       ThirdOrderSparse=true,profile=false,
                       kwargs...)

    if ThirdOrderSparse
        A = load_ThirdOrderSymTensor(graph_A_file)
        B = load_ThirdOrderSymTensor(graph_B_file)
    else
        A = load(graph_A_file,false,"COOTen")
        B = load(graph_B_file,false,"COOTen")
    end


    if profile
        return align_tensors_profiled(A,B;kwargs...)
    else
        return align_tensors(A,B;kwargs...)
    end
end

