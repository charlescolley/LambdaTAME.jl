abstract type AlignmentMethod end
struct ΛTAME_M <: AlignmentMethod end
struct ΛTAME_MultiMotif_M <: AlignmentMethod end
struct LowRankTAME_M <: AlignmentMethod end
struct TAME_M <: AlignmentMethod end
struct EigenAlign_M <: AlignmentMethod end
struct LowRankEigenAlign_M <: AlignmentMethod end
struct Degree_M <: AlignmentMethod end
struct Random_M <: AlignmentMethod end

abstract type RandomGraphType end 
struct ErdosRenyi <: RandomGraphType end 
struct RandomGeometric <: RandomGraphType end 
struct HyperKron <: RandomGraphType end

#=------------------------------------------------------------------------------
                        Formatting Routines
------------------------------------------------------------------------------=#
function produce_ssten_from_triangles(file;use_metis=false,lcc=false)

    A = MatrixNetworks.readSMAT(file)
    (n,m) = size(A)
    if(n != m)
        println("rectangular")
    end

	if !issymmetric(A)
		A = max.(A,A')  #symmetrize for Triangles routine
    end

    if use_lcc
        A, _ = largest_component(A)
    end

	if use_metis
		apply_Metis_permutation!(A)
	end

    T = collect(MatrixNetworks.triangles(A))

	alterfilename = (file,postfix)-> split(file,".smat")[1]*postfix
	if use_metis
	    output_file = alterfilename(file,"_with_metis.ssten")
	else
	    output_file = alterfilename(file,".ssten")
	end

    open(output_file,"w") do f
        write(f,"$(3)\t$(n)\t$(length(T))\n")

        for (v_i,v_j,v_k) in T
            write(f,"$(v_i)\t$(v_j)\t$(v_k)\t1.0\n")
        end
    end

end

function load_UnweightedThirdOrderSymTensor(filepath;enforceFormatting = true)

	#check path validity
	@assert filepath[end-5:end] == ".ssten"

    open(filepath) do file
		#preallocate from the header line
		order, n, m =
			[parse(Int,elem) for elem in split(chomp(readline(file)),'\t')]
        @assert order == 3

		Ti = [ Vector{Tuple{Int,Int}}(undef, 0) for i in 1:n ]

		i = 1
		@inbounds for line in eachline(file)
			entries = split(chomp(line),'\t')

			if enforceFormatting
				(ti,tj,tk) = sort([parse(Int,elem) for elem in entries[1:end-1]])
			else
				(ti,tj,tk) = [parse(Int,elem) for elem in entries[1:end-1]]
			end

			if 0 == ti || 0 == tj || 0 == tk
				error("elements must be indexed by 1.")
			end
			push!(Ti[ti], (tj,tk))
			push!(Ti[tj], (ti,tk))
			push!(Ti[tk], (ti,tj))

		end

		sort!.(Ti)
		return UnweightedThirdOrderSymTensor(n,Ti)
	end

end
"""------------------------------------------------------------------------------
  Loads in a ThirdOrderSymTensor from an ssten file. Data specifications can be 
  found in the 'formatting_specification.info' file the data/ folder. The 
  enforceFormatting can be used to ensure that the indices are sorted in 
  increasing order, and any files which have 0's in indices are updated to be
  indexed by 1.
------------------------------------------------------------------------------"""
function load_ThirdOrderSymTensor(filepath;enforceFormatting = true)

	#check path validity
	@assert filepath[end-5:end] == ".ssten"

    open(filepath) do file
		#preallocate from the header line
		order, n, m =
			[parse(Int,elem) for elem in split(chomp(readline(file)),'\t')]
        @assert order == 3

		indices = Array{Int,2}(undef,m,order)
		values = Array{Float64,1}(undef,m)


		i = 1
		@inbounds for line in eachline(file)
			entries = split(chomp(line),'\t')

			indices[i,:] = [parse(Int,elem) for elem in entries[1:end-1]]
			if enforceFormatting
				sort!(indices[i,:])
			end
			values[i] = parse(Float64,entries[end])
			i += 1
		end

		#check for 0 indexing
		zero_indexed = false

		@inbounds for i in 1:m
		    if indices[i,1] == 0
    			zero_indexed = true
				break
			end
	    end


		if zero_indexed
			indices .+= 1
		end

        return ThirdOrderSymTensor(n,indices,values)
    end
end

function write_to_armadillo_raw_ascii_format(X::Array{T,2},output_file::String) where T
    open(output_file,"w") do f
        for row in eachrow(X)
            println(f,join(row," "))
        end
    end
end
#=------------------------------------------------------------------------------
                        Local File Experiments Routines
------------------------------------------------------------------------------=#

#TODO: update to use kwargs properly
function self_alignment(ssten_files::Array{String,1};method="LambdaTAME")

    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files))

    for f in ssten_files
        A = load(f;type="COOTen")

        #permute tensor to align against
        p = sort(1:A.cubical_dimension,by=i->rand())
        A_permuted = COOTen(A.indices,A.vals,A.cubical_dimension)
        permute_tensor!(A_permuted,p)

        best_score, _, _ = align_tensors(A,A_permuted;method=method)

        println("$f aligned with $best_score")
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


function apply_Metis_permutation!(A::SparseMatrixCSC,k::Int=100)

	n = size(A,1)
	metis_partition = Metis.Metis.partition(A, 100)
	p = sort(1:n, by= i-> metis_partition[i])

	permute!(A,p,p)
end

#TODO: put in kwargs
function pairwise_alignment(dir)

    #align all .ssten files
    ssten_files = sort([f for f in readdir(dir) if occursin(".ssten",f)])

    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files),length(ssten_files))

    for i in 1:length(ssten_files)
        for j in i+1:length(ssten_files)
            Best_alignment_ratio[i,j] = align_from_files(dir*ssten_files[i],dir*ssten_files[j])
            Best_alignment_ratio[i,j] = Best_alignment_ratio[j,i]
        end
    end

    return ssten_files, Best_alignment_ratio
end

#=
function distributed_pairwise_alignment(dir::String;kwargs...)

    #align all .ssten files
    ssten_files = sort([f for f in readdir(dir) if occursin(".ssten",f)])
    distributed_pairwise_alignment(ssten_files,dir;kwargs...)

end
=#


"""------------------------------------------------------------------------------
  This function runs alignemnt experiments in parallel using the @spawn macro of 
  Distributed.jl. This function should be run in the src file, as the @everywhere
  macro will only work properly there. Any additional kwargs are passed to 
  'align_matrices' or 'align_tensors' depending on the file extensions.

  Inputs
  ------
  * files - (Array{String,1}):
    List of the file names. Files may be either all SMAT or all STEN files. 
    'align_tensors' or 'align_matrices' will be called depending on the extension. 
  * dirpath - (String):
    The path to where the files exist. 
  * method - (String):
    What type of alignment method to run, options are 'LambdaTAME', 'TAME',
    'LowRankTAME', 'EigenAlign', 'LowRankEigenAlign', 'Degree', or 'Random'. 
  * profile - (Bool):
    Whether or not to profile the algorithm. Setting this flag to true will 
    record different things depending on the algorithm, but will always return 
    runtime information. 
Output
------
  * 'exp_results' - (type omitted):
    The final array of results for the series of experiments, the size of each 
    entry will differ depending on whether or not profile was True. At the very
    least, each entry will contain the the pair of file names aligned, the 
    number of triangles matched in the alignemnt, and the maximum number of 
    triangles possible to align (the minimum number of triangles in either 
    network). 

------------------------------------------------------------------------------"""
function distributed_pairwise_alignment(files::Array{String,1},dirpath::String;
                                        method="LambdaTAME",profile=false,kwargs...)

    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")

    alignment_object = ""
    if all( [f[end-5:end] == ".ssten" for f in files]) #tensors 
        alignment_object = "Tensors" 
    elseif all( [f[end-4:end] == ".smat" for f in files]) #matrices
        alignment_object = "Matrices" 
    else 
        throw(ArgumentError("all files must be the same file type, either all '.ssten' or '.smat'."))
    end

    futures = []
	exp_results = []

#    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files),length(ssten_files))

    for i in 1:length(files)
        for j in i+1:length(files)

            if alignment_object == "Tensors"
                future = @spawn align_tensors(dirpath*files[i],dirpath*files[j];profile=profile,method=method,kwargs...)
            elseif alignment_object == "Matrices"
                A = MatrixNetworks.readSMAT(dirpath*files[i])
                B = MatrixNetworks.readSMAT(dirpath*files[j])
                future = @spawn align_matrices(A,B;profile=profile,method=method,kwargs...)
            end
            push!(futures,((i,j),future))
        end
    end

    for ((i,j), future) in futures

		if method == "LambdaTAME" ||  method == "LowRankTAME"
			if profile
				matched_tris, max_tris, _, _, best_matching, results = fetch(future)
			else
				matched_tris, max_tris, _, _, best_matching = fetch(future)
			end
		elseif method == "TAME"
			if profile
				matched_tris, max_tris, _, best_matching, results = fetch(future)
			else
				matched_tris, max_tris, _, best_matching = fetch(future)
            end
        elseif method == "EigenAlign" || method == "Degree" || method == "Random" || method == "LowRankEigenAlign"
            A_tens, B_tens, matched_tris, best_matching, results = fetch(future)
            max_tris = min(A_tens, B_tens)
            profile = true
		end
		if profile
			push!(exp_results,(files[i],files[j],matched_tris, max_tris, best_matching, results))
		else
			push!(exp_results,(files[i],files[j],matched_tris, max_tris, best_matching))
		end
    end

    return exp_results
end


function self_alignment(dir::String;kwargs...)

    #align all .ssten files
    ssten_files = sort([dir*"/"*f for f in readdir(dir) if occursin(".ssten",f)])
    return self_alignment(ssten_files;kwargs...)

end

#=------------------------------------------------------------------------------
                      Generated Graph Experiments Routines
------------------------------------------------------------------------------=#
"""------------------------------------------------------------------------------
  This function runs randomly generated experiments in parallel using the @spawn 
  macro of Distributed.jl. This function should be run in the src file, as the 
  @everywhere macro will only work properly there. Any additional kwargs are 
  passed to 'random_graph_exp'.

  Inputs
  ------
  * 'trial_count' - (Int):
    The number of trials to run for each of the experiments conducted. 
  * 'seed_exps' - (Bool):
    Indicates whether or not to seed the experiments. Seeds are generated by 
    seeding Random.seed!(0), and then generating an array of UInts using the 
    number of 'p_remove' and 'n_sizes' variables, and the 'trial_count'. Changing 
    these may lead to different seeds. 
  * 'n_sizes' - (Array{Int,1}):
    The different sizes of graphs to generate experiments for. 
  * 'p_remove' - (Array{Float,1}):
    The different probability of edge removals to test in the ER noise models. 
  * profile - (Bool):
    Whether or not to profile the algorithm. Setting this flag to true will 
    record different things depending on the algorithm, but will always return 
    runtime information. 
Output
------
  * 'exp_results' - (type omitted):
    The final array of results for the series of experiments, the size of each 
    entry will differ depending on whether or not profile was True. At the very
    least, the each entry will contain the seed used, the 'p_remove' value, the 
    number of vertices in the graph, the ground truth accuracy of the alignment, 
    the degree weighted accuracy of the alignment, the number of matched 
    triangles, the number of triangles each graph, and the maximum number of 
    triangles possible to align. Anything additional returned by the functions 
    with the profile flag on, is appended at the end.     
------------------------------------------------------------------------------"""
function distributed_random_trials(trial_count::Int,seed_exps::Bool=false
                                   ;method::AlignmentMethod=ΛTAME(),graph::RandomGraphType=ErdosRenyi(),
								   n_sizes = [100, 500, 1000, 2000,5000], p_remove = [.01,.05],
								   profile=false,kwargs...)

    #only handling even batch sizes
    #@assert trial_count % process_count == 0

    #ensure file is loaded on all processes
    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")

   
	if seed_exps
		Random.seed!(0)
		seeds = rand(UInt64, length(p_remove),length(n_sizes), trial_count)
	end

	seed = nothing
	results = []
    p_index = 1

    futures = []


    for p in p_remove

        n_index = 1

        for n in n_sizes

            for trial = 1:trial_count
  
                if seed_exps
                    seed = seeds[p_index,n_index,trial]
                end
                println(seed)
                future = @spawn random_graph_exp(n,p,graph;
                                                    profile=profile,seed=seed,method=method,
                                                    kwargs...)
                push!(futures,(seed,p,n,future))

            end

            n_index += 1
        end
        p_index += 1
    end


    for (seed,p,n,future) in futures
        if profile 
            if typeof(method) === ΛTAME_M ||  typeof(method) === LowRankTAME_M
                d_A, d_B, perm, (A_tris, B_tris,(matched_tris, max_tris, _, _,best_matching, exp_results))= fetch(future)
            elseif typeof(method) == TAME_M
                d_A, d_B, perm, (A_tris, B_tris,(matched_tris, max_tris, _, best_matching, exp_results))= fetch(future)
            end

            accuracy = sum([1 for (i,j) in enumerate(perm) if get(best_matching,j,-1) == i])/n
            D_A = sum(d_A)
            D_B = sum(d_B)
            degree_weighted_accuracy = sum([(get(best_matching,j,-1) == i) ? ((d_A[i] + d_B[j])/(D_A+D_B)) : 0.0 for (i,j) in enumerate(perm)])
      
            push!(results,(seed, p, n, accuracy, degree_weighted_accuracy, matched_tris, A_tris, B_tris, max_tris, exp_results))
        else
            if typeof(method) === ΛTAME_M ||  typeof(method) === LowRankTAME_M
                d_A, d_B, perm, (A_tris, B_tris,(matched_tris, max_tris, _, _, best_matching)) = fetch(future)
            elseif typeof(method) === ΛTAME_MultiMotif_M
                d_A, d_B, perm, (best_matching_score, max_motif_match,best_matched_motifs, _, _, best_matching) = fetch(future)
            elseif typeof(method) === TAME_M
                d_A, d_B, perm, (A_tris, B_tris, (matched_tris, max_tris, _, best_matching)) = fetch(future)
            elseif typeof(method) === EigenAlign_M || typeof(method) == Degree_M || typeof(method) === Random_M || typeof(method) === LowRankEigenAlign_M
                d_A, d_B, perm, (A_tris, B_tris, matched_tris, best_matching, _) = fetch(future)
                max_tris = minimum((A_tris,B_tris))
            end

            accuracy = sum([1 for (i,j) in enumerate(perm) if get(best_matching,j,-1) == i])/n
            D_A = sum(d_A)
            D_B = sum(d_B)
            degree_weighted_accuracy = sum([(get(best_matching,j,-1) == i) ? ((d_A[i] + d_B[j])/(D_A+D_B)) : 0.0 for (i,j) in enumerate(perm)])
      
            if typeof(method) === ΛTAME_MultiMotif_M
                push!(results,( seed, p, n, accuracy, degree_weighted_accuracy, best_matching_score, max_motif_match, best_matched_motifs))
            else
                push!(results,( seed, p, n, accuracy, degree_weighted_accuracy, matched_tris, A_tris, B_tris, max_tris))
            end
        end                    
    end

    return results

end


"""-----------------------------------------------------------------------------
  Runs an instance of a graph alignment problem using a random graph model. Once 
  a graph is generated, we use the function 'ER_noise_model' to generate a 
  second network to align against it. Once the second network is generated it is 
  randomly permuted. All additional kwargs are passed to the 'align_matrices' 
  function.

  Inputs
  ------
  * n - (Int):
    The node of nodes to use in each network. 
  * 'p_remove' -(Float):
    The probability of removing an edge in the ER noise model. The probability 
    of adding in an edge is computed using a function of this parameter at the 
    density of the generated network. 
  * 'graph_type' - (String):
    The type of graph used to generate the network. Options include 'ER' for an
    Erdos Renyi network, 'RandomGeometric' for a Random Geometric network from 
    a 2D unit square, and 'HyperKron' which comes from the Triangle Generalized 
    Preferential Attachment model. 
  * seed - (Any):
    A seed used to replicate the results. Our experiments typically use UInts.
  * 'use_metis' - (Bool):
    Indicates whether or not to use metis clusters to permute the adjacency 
    matrix. For large networks this can improve the runtime by improving the 
    access patterns. 
  * degreedist - (UnivariateDistribution):
    A random variable generator from Distributions.jl which is used to generate
    the degrees for random nodes. We primarily use 
    Distributions.LogNormal(log(5,1)). 
  * 'p_edges' - (func Int - Float):
    An anonymous function used for computing the probability of adding edges. 
    Primarily used in the ER graphs for generating the probability of adding 
    in edges from the number of nodes in the network. an example expected input
    would be 'n -> 2*log(n)/n'. 

    Outputs
    -------
    * 'd_A','d_B' - (Array{Float,1}):
      The degrees of all the nodes in graphs A and B. The ith entry of each 
      being the degree of node i in the respective networks. 
    * perm - (Array{Int,1}):
      The permutation used to shuffle the nodes in graph B. Used for computing 
      the precision of the methods. 
    * All additional return variables are from 'align_matrices' function call. 
-----------------------------------------------------------------------------"""
function random_graph_exp(n::Int, p_remove::Float64,graph::RandomGraphType;
                          seed=nothing,use_metis=false,degreedist=nothing,p_edges=nothing,kwargs...)

	if seed !== nothing
		Random.seed!(seed)
	end

    if typeof(graph) === ErdosRenyi
        if degreedist === nothing
            if p_edges === nothing 
                p = 2*log(n)/n
            else
                p = p_edges(n)
            end
            A = erdos_renyi(n,p)
        else
            println(degreedist)
            A = erdos_renyi(n;degreedist)
            p = nnz(A)/n^2
        end
        
	elseif typeof(graph) === RandomGeometric

		if degreedist === nothing
            k = 10
            if p_edges === nothing 
                p = k/n
            end
			A = random_geometric_graph(n,k)
		else
			d = 2
			A = spatial_network(n, d;degreedist= degreedist)
			p = nnz(A)/n^2
		end

    elseif typeof(graph) == HyperKron
        if p_edges === nothing 
            p = .4#2*log(n)/n
        end
		r = .4

		A = sparse(gpa_graph(n,p,r,5))

	else
		error("invalid graph type: $graph_type\n must be either 'ER','RandomGeometric' or 'HyperKron'")
	end

    p_add = (p*p_remove)/(1-p)
	B = ER_noise_model(A,p_remove,p_add)

    perm = shuffle(1:n)
    B = B[perm,perm]

	if use_metis
		apply_Metis_permutation!(A,100)
		apply_Metis_permutation!(B,100)
	end

    d_A = A*ones(n)
    d_B = B*ones(n)

    return d_A,d_B,perm,align_matrices(A,B;kwargs...)
   
end

function ER_noise_model(A,p_remove::F,p_add::F) where {F <: AbstractFloat}
    n = size(A,1)
    B = copy(A)

    is,js,_ = findnz(erdos_renyi(n,p_remove))
    for (i,j) in zip(is,js)
        if A[i,j] > 0
            B[i,j] = 0
            B[j,i] = 0
        end
    end

    is,js,_ = findnz(erdos_renyi(n,p_add))
    for (i,j) in zip(is,js)
        if A[i,j] == 0.0
            B[i,j] = 1
            B[j,i] = 1
        end
    end

    dropzeros!(B)
    return B
end

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
function align_matrices(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{T,Int};profile=false,kwargs...) where T

    A_ten = graph_to_ThirdOrderTensor(A)
    B_ten = graph_to_ThirdOrderTensor(B)
    
    method = typeof(kwargs[:method])

    if method === ΛTAME_M || method === LowRankTAME_M  || method === TAME_M
        if profile
            return size(A_ten.indices,1), size(B_ten.indices,1), align_tensors_profiled(A_ten,B_ten;kwargs...)
        else
            return size(A_ten.indices,1), size(B_ten.indices,1), align_tensors(A_ten,B_ten;kwargs...)
        end
    elseif method === ΛTAME_MultiMotif_M

        A_tensors = tensors_from_graph(A,kwargs[:orders],kwargs[:samples])        
        B_tensors = tensors_from_graph(B,kwargs[:orders],kwargs[:samples])

        subkwargs = Dict([(k,v) for (k,v) in kwargs if k != :orders && k != :samples])
        return align_tensors(A_tensors,B_tensors;subkwargs...)

    elseif method === EigenAlign_M || method === Degree_M || method === Random_M || method === LowRankEigenAlign_M
        
        if method === LowRankEigenAlign_M
            iters = 10
            (ma,mb,_,_),t = @timed align_networks_eigenalign(A,B,iters,"lowrank_svd_union",3)
            matching = Dict{Int,Int}([i=>j for (i,j) in zip(ma,mb)]) 
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
        triangle_count, gaped_triangles, _ = TAME_score(A_ten,B_ten,matching) 
        return size(A_ten.indices,1), size(B_ten.indices,1), triangle_count, matching, t 
    else
        throw(ArgumentError("method must be of type LambdaTAME_M, LowRankTAME_M, TAME_M, EigenAlign_M, LowRankEigenAlign_M, Degree_M, or Random_M."))
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
        println("Symmetrizing matrix")
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


# --------------------------------------------------------------------------- #
#						       Random Graph Models
# --------------------------------------------------------------------------- #

function erdos_renyi(n,p)
    A = sprand(n,n,p)
    U = triu(A,1)
    is,js,_ = findnz(max.(U,U'))
    return sparse(is,js,ones(length(is)),n,n)
end

function erdos_renyi(n;degreedist=LogNormal(log(5),1))

    println("made it")
    # form the edges for sparse
    ei = Int[]
    ej = Int[]

    for i=1:n
      deg = ceil(Int,minimum((rand(degreedist),n-1)))
      neighbors = sample(1:n,deg+1, replace = false)

      for j in neighbors
        if i != j
          push!(ei,i)
          push!(ej,j)
        end
      end
    end

    A = sparse(ei,ej,1.0,n,n)

    return max.(A,A')
end

#--                             David's graph code                          --#

function random_geometric_graph(n,k)
  xy = rand(2,n)
  T = BallTree(xy)
  idxs = knn(T, xy, minimum((k,n)))[1]
  # form the edges for sparse
  ei = Int[]
  ej = Int[]
  for i=1:n
    for j=idxs[i]
      if i != j
        push!(ei,i)
        push!(ej,j)
      end
    end
  end
  A = sparse(ei,ej,1.0,n,n)

  return max.(A,A')
end

function spatial_network(n::Integer, d::Integer; degreedist=LogNormal(log(5),1))
  xy, ei, ej = spatial_graph_edges(n, d;degreedist=degreedist)
  A = sparse(ei,ej,1.0,n,n)
  return max.(A,A')
end

function spatial_graph_edges(n::Integer,d::Integer;degreedist=LogNormal(log(5),1))
  xy = rand(d,n)
  T = BallTree(xy)
  # form the edges for sparse
  ei = Int[]
  ej = Int[]
  for i=1:n
    deg = ceil(Int,minimum((rand(degreedist),n-2)))
    idxs, dists = knn(T, xy[:,i], deg+1)
    for j in idxs
      if i != j
        push!(ei,i)
        push!(ej,j)
      end
    end
  end
  return xy, ei, ej
end

