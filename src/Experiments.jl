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

abstract type RandomGraphType end 
struct ErdosRenyi <: RandomGraphType end 
struct RandomGeometric <: RandomGraphType end 
struct HyperKron <: RandomGraphType end

abstract type NoiseModelFlag end 
struct ErdosRenyiNoise  <: NoiseModelFlag end 
struct DuplicationNoise <: NoiseModelFlag end 

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
                                        method::AlignmentMethod=ΛTAME_M(),profile=false,kwargs...)

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

        if method === ΛTAME_M() && kwargs[:matchingMethod] === ΛTAME_GramMatching()
            if profile
                matched_tris, max_tris, best_matching, results = fetch(future)
			else 
				matched_tris, max_tris, best_matching = fetch(future)
			end
        elseif method === LowRankTAME_M() || (method === ΛTAME_M() && kwargs[:matchingMethod] === ΛTAME_rankOneMatching())
			if profile
				matched_tris, max_tris, _, _, best_matching, results = fetch(future)
			else
				matched_tris, max_tris, _, _, best_matching = fetch(future)
			end
		elseif method === TAME_M()
			if profile
				matched_tris, max_tris, _, best_matching, results = fetch(future)
			else
				matched_tris, max_tris, _, best_matching = fetch(future)
            end
        elseif method == EigenAlign_M() || method == Degree_M() || method == Random_M() || method == LowRankEigenAlign_M()
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
function distributed_random_trials(trial_count::Int,noise_model::ErdosRenyiNoise,seed_exps::Bool=false
                                   ;method::AlignmentMethod=ΛTAME(),graph::RandomGraphType=ErdosRenyi(),
                                   n_sizes = [100, 500, 1000, 2000,5000],p_remove = [.01,.05],
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
                future = @spawn random_graph_exp(n,p,graph; profile=profile,seed=seed,
                                                 method=method,noise_model=noise_model,
                                                 kwargs...)
                push!(futures,(seed,p,n,future))

            end

            n_index += 1
        end
        p_index += 1
    end

    #TODO: fix parsing of returned functions
    for (seed,p,n,future) in futures
        if profile 
         
            if method === ΛTAME_M() && kwargs[:matchingMethod] === ΛTAME_GramMatching()
                d_A, d_B, perm, (A_tris, B_tris,(matched_tris, max_tris,best_matching, exp_results)) = fetch(future)
            elseif (method === ΛTAME_M() && kwargs[:matchingMethod] === ΛTAME_rankOneMatching()) ||  method === LowRankTAME_M()
                d_A, d_B, perm, (A_tris, B_tris,(matched_tris, max_tris, _, _,best_matching, exp_results))= fetch(future)
            elseif method === TAME_M()
                d_A, d_B, perm, (A_tris, B_tris,(matched_tris, max_tris, _, best_matching, exp_results))= fetch(future)
            end

            accuracy_B_to_A = sum([1 for (i,j) in enumerate(perm) if get(best_matching,j,-1) == i])/n
            accuracy_A_to_B = sum([1 for (i,j) in enumerate(perm) if get(best_matching,i,-1) == j])/n
            accuracy = max(accuracy_B_to_A,accuracy_A_to_B) 

            D_A = sum(d_A)
            D_B = sum(d_B)
            degree_weighted_accuracy = sum([(get(best_matching,j,-1) == i) ? ((d_A[i] + d_B[j])/(D_A+D_B)) : 0.0 for (i,j) in enumerate(perm)])
            
            push!(results,(seed, p, n, accuracy, degree_weighted_accuracy, matched_tris, A_tris, B_tris, max_tris, exp_results))

        else
            if method === ΛTAME_M() && kwargs[:matchingMethod] === ΛTAME_GramMatching()
                d_A, d_B, perm, (A_tris, B_tris,(matched_tris, max_tris,best_matching)) = fetch(future)
            elseif (method === ΛTAME_M() && kwargs[:matchingMethod] === ΛTAME_rankOneMatching()) ||  method === LowRankTAME_M
                d_A, d_B, perm, (A_tris, B_tris,(matched_tris, max_tris, _, _, best_matching)) = fetch(future)
            elseif method === ΛTAME_MultiMotif_M()
                if kwargs[:matchingMethod] === ΛTAME_GramMatching()
                    d_A, d_B, perm, (A_motifCounts, B_motifCounts,best_matching_score, max_motif_match, best_matched_motifs, best_matching) = fetch(future)
                else
                    d_A, d_B, perm, (A_motifCounts, B_motifCounts,best_matching_score, max_motif_match,best_matched_motifs, _, _, best_matching) = fetch(future)
                end
            elseif method === TAME_M()
                d_A, d_B, perm, (A_tris, B_tris, (matched_tris, max_tris, _, best_matching)) = fetch(future)
            elseif method === EigenAlign_M() || method == Degree_M() || method === Random_M() || method === LowRankEigenAlign_M() || method === LowRankEigenAlignOnlyEdges_M()
                d_A, d_B, perm, (A_tris, B_tris, matched_tris, best_matching, _) = fetch(future)
                max_tris = minimum((A_tris,B_tris))
            end

            accuracy_B_to_A = sum([1 for (i,j) in enumerate(perm) if get(best_matching,j,-1) == i])/n
            accuracy_A_to_B = sum([1 for (i,j) in enumerate(perm) if get(best_matching,i,-1) == j])/n

            accuracy = max(accuracy_B_to_A,accuracy_A_to_B)
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

function distributed_random_trials(trial_count::Int,noise_model::DuplicationNoise,seed_exps::Bool=false;
        method::AlignmentMethod=ΛTAME(),graph::RandomGraphType=ErdosRenyi(),
        n_sizes = [100, 500, 1000, 2000,5000],edge_inclusion_p = [.5,1.0],
        step_percentage=[.1],profile=false,kwargs...)

    #only handling even batch sizes
    #@assert trial_count % process_count == 0

    #ensure file is loaded on all processes
    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")
    

    if seed_exps
        Random.seed!(0)
        seeds = rand(UInt64, length(step_percentage),length(edge_inclusion_p),length(n_sizes), trial_count)
    end

    seed = nothing
    results = []

    step_index = 1
    p_index = 1

    futures = []

    for sp in step_percentage

        p_index = 1
        for p in edge_inclusion_p

            n_index = 1

            for n in n_sizes

                for trial = 1:trial_count

                    if seed_exps
                        seed = seeds[step_index,p_index,n_index,trial]
                    end
                    future = @spawn random_graph_exp(n,p,graph; profile=profile,noise_model=noise_model,step_percent=sp,
                                                     seed=seed,method=method, kwargs...)
                    push!(futures,(seed,p,n,sp,future))
                end
                n_index += 1
            end
            p_index += 1
        end
        step_index += 1
    end

    #TODO: fix parsing of returned functions
    for (seed,p,n,sp,future) in futures
        if profile 
            
            if method === ΛTAME_M() && kwargs[:matchingMethod] === ΛTAME_GramMatching()
                perm,dup_vertices, (A_tris, B_tris,(matched_tris, max_tris,best_matching, exp_results)) = fetch(future)
            elseif (method === ΛTAME_M && kwargs[:matchingMethod] === ΛTAME_rankOneMatching()) ||  method === LowRankTAME_M()
                perm,dup_vertices, (A_tris, B_tris,(matched_tris, max_tris, _, _,best_matching, exp_results))= fetch(future)
            elseif method === TAME_M
                perm,dup_vertices, (A_tris, B_tris,(matched_tris, max_tris, _, best_matching, exp_results))= fetch(future)
            end

            accuracy_B_to_A = sum([1 for (i,j) in enumerate(perm) if get(best_matching,j,-1) == i])/n
            accuracy_A_to_B = sum([1 for (i,j) in enumerate(perm) if get(best_matching,i,-1) == j])/n
            #TODO: use a more robust method of catching how matchings are oriented
            accuracy = max(accuracy_B_to_A,accuracy_A_to_B) 

            dup_tolerant_perm = replaced_duplications_with_originals(perm,n, dup_vertices)
            DT_accuracy_B_to_A = sum([1 for (i,j) in enumerate(dup_tolerant_perm) if get(best_matching,j,-1) == i])/n
            DT_accuracy_A_to_B = sum([1 for (i,j) in enumerate(dup_tolerant_perm) if get(best_matching,i,-1) == j])/n
            dup_vertex_tolerant_accuracy = max(DT_accuracy_B_to_A,DT_accuracy_A_to_B) 
            
            push!(results,(seed, p, n, sp, accuracy, dup_vertex_tolerant_accuracy, matched_tris, A_tris, B_tris, max_tris, exp_results))

        else

            if method === ΛTAME_M() && kwargs[:matchingMethod] === ΛTAME_GramMatching()
                perm, dup_vertices, (A_tris, B_tris,(matched_tris, max_tris,best_matching)) = fetch(future)
            elseif (method === ΛTAME_M() && kwargs[:matchingMethod] === ΛTAME_rankOneMatching()) ||  method === LowRankTAME_M()
                perm, dup_vertices, (A_tris, B_tris,(matched_tris, max_tris, _, _,best_matching))= fetch(future)
            elseif method === TAME_M()
                perm, dup_vertices, (A_tris, B_tris,(matched_tris, max_tris, _, best_matching))= fetch(future)
            elseif method === ΛTAME_MultiMotif_M()
 
                if kwargs[:matchingMethod] === ΛTAME_GramMatching()
                    #TODO: fix naming conventions
                    perm, dup_vertices, (A_motifCounts, B_motifCounts, best_matching_score, max_motif_match, best_matched_motifs, best_matching) = fetch(future)
                else
                    perm, dup_vertices, (A_motifCounts, B_motifCounts, best_matching_score, max_motif_match, best_matched_motifs, _, _, best_matching) = fetch(future)
                end
                A_tris = -1
                B_tris = -1
            elseif method === EigenAlign_M() || method == Degree_M() || method === Random_M() || method === LowRankEigenAlign_M() || method === LowRankEigenAlignOnlyEdges_M()
                perm, dup_vertices, (A_tris, B_tris, matched_tris, best_matching, _) = fetch(future)
                max_tris = minimum((A_tris,B_tris))
            end

            accuracy_B_to_A = sum([1 for (i,j) in enumerate(perm) if get(best_matching,j,-1) == i])/n
            accuracy_A_to_B = sum([1 for (i,j) in enumerate(perm) if get(best_matching,i,-1) == j])/n
            #TODO: use a more robust method of catching how matchings are oriented
            accuracy = max(accuracy_B_to_A,accuracy_A_to_B) 

            dup_tolerant_perm = replaced_duplications_with_originals(perm,n, dup_vertices)
            DT_accuracy_B_to_A = sum([1 for (i,j) in enumerate(dup_tolerant_perm) if get(best_matching,j,-1) == i])/n
            DT_accuracy_A_to_B = sum([1 for (i,j) in enumerate(dup_tolerant_perm) if get(best_matching,i,-1) == j])/n
            dup_vertex_tolerant_accuracy = max(DT_accuracy_B_to_A,DT_accuracy_A_to_B) 
            

            if typeof(method) === ΛTAME_MultiMotif_M
                push!(results,( seed, p, n, sp, accuracy, dup_vertex_tolerant_accuracy, best_matching_score,A_motifCounts, B_motifCounts, best_matched_motifs))
            else
                push!(results,( seed, p, n, sp, accuracy, dup_vertex_tolerant_accuracy, matched_tris, A_tris, B_tris, max_tris))
            end

        end                    
    end    
    return results
end

#used as a helper function for mapping duplicated vertices back to vertices in original graph 
function replaced_duplications_with_originals(perm,n, dup_vertices)
    dup_tolerant_perm = copy(perm)
    dup_vertex_mapping = Dict([(n+i,v) for (i,v) in enumerate(dup_vertices)])

    while maximum(dup_tolerant_perm) > n
        dup_tolerant_perm = [haskey(dup_vertex_mapping,i) ? dup_vertex_mapping[i] : i for i in dup_tolerant_perm]
    end

    return dup_tolerant_perm

end

"""-----------------------------------------------------------------------------
                        ΛTAME Matching Experiments
-----------------------------------------------------------------------------"""
function ΛTAME_matching_exp(A_file::String, B_file::String, args...)

    @assert A_file[end-5:end] == ".ssten"
    @assert B_file[end-5:end] == ".ssten"

    A = load_SymTensorUnweighted(A_file,'\t') #delimiter may need to be updated
    B = load_SymTensorUnweighted(B_file,'\t')

    return ΛTAME_matching_exp(A,B,args...)

end

function ΛTAME_matching_exp(A_file::String, B_file::String, output_file::String=nothing, args...)

    @assert A_file[end-5:end] == ".ssten"
    @assert B_file[end-5:end] == ".ssten"
    @assert output_file[end-4:end] == ".json"

    A = load_SymTensorUnweighted(A_file,'\t') #delimiter may need to be updated
    B = load_SymTensorUnweighted(B_file,'\t')

    open(output_file,"w") do f
        JSON.print(f,ΛTAME_matching_exp(A,B,args...))
    end

end

function ΛTAME_matching_exp(A::Union{ThirdOrderSymTensor,SymTensorUnweighted{S}},
                            B::Union{ThirdOrderSymTensor,SymTensorUnweighted{S}},
                            max_iter::Int,alphas::Array{Float64,1}=[.5,1.0],
                            betas::Array{Float64,1}=[0.0,1.0,10.0,100.0]) where {S <:Motif}

    tol = 1e-16 # we want to track how triangles change over iterations

    if typeof(A) == ThirdOrderSymTensor
        A_motifs = size(A.indices,1)
        B_motifs = size(B.indices,1)
    else
        A_motifs= size(A.indices,2)
        B_motifs= size(B.indices,2)
    end

    results = []
    for α in alphas
        for β in betas

            matching_counts = []
            U,V = ΛTAME(A, B, β,max_iter,tol,α)

            d = size(U,2)

            for i=1:d
                motifs_matched,_,_ = TAME_score(A,B,U[:,1:i]*V[:,1:i]')
                push!(matching_counts,motifs_matched)
            end

            push!(results,(α,β,matching_counts))
        end
    end
    
    A_motifs, B_motifs, results
end


"""-----------------------------------------------------------------------------
                              SSHOPM Experiments
-----------------------------------------------------------------------------"""
function distributed_SSHOPM_exps(tensor_files::Array{String,1},output_path::String,seed::Bool;kwargs...)

    #TODO: test
    seeds = Array{UInt,2}(undef,1,1) 

    if seed
        seeds = rand(UInt,length(tensor_files),length(tensor_files))
    end

    futures = []
    for i =1:length(tensor_files)
        for j=i+1:length(tensor_files)

            future = @spawn SSHOPM_exp(tensor_files[i],tensor_files[j],output_path,seeds[i,j])
            push!(futures,(tensor_files[i],tensor_files[j],future))
        end
    end

    #TODO: join using a sigpoll equivilant
    for (ten_A, ten_B, future) in futures
        fetch(future)
        println("joined experiment $ten_A + $ten_B")
    end
end

function distributed_SSHOPM_exps(tensor_files::Array{Tuple{String,String},1},output_path::String,seed::Bool;kwargs...)

    #TODO: test
    seeds = Array{UInt,2}(undef,1,1) 

    if seed
        seeds = rand(UInt,length(tensor_files),length(tensor_files))
    end

    futures = []
    for (tensor_A_file, tensor_B_file) in tensor_files

        future = @spawn SSHOPM_exp(tensor_A_file, tensor_B_file, output_path,seeds[i,j])
        push!(futures,(tensor_A_file, tensor_B_file,future))

    end

    #TODO: join using a sigpoll equivilant
    for (ten_A, ten_B, future) in futures
        fetch(future)
        println("joined experiment $ten_A + $ten_B")
    end
end


function SSHOPM_exp(tensor_A_file::String, tensor_B_file::String,output_path::String,seed::UInt)

    @assert tensor_A_file[end-5:end] == ".ssten"
    @assert tensor_B_file[end-5:end] == ".ssten"
    @assert output_path[end] == '/'

    A = load_ThirdOrderSymTensor(tensor_A_file)
    B = load_ThirdOrderSymTensor(tensor_B_file)

    A_root = split(split(tensor_A_file,"/")[end],".ssten")[1]
    B_root = split(split(tensor_B_file,"/")[end],".ssten")[1]
    
    tol = 1e-16
    max_iter = 30 
    β = 0.0
    samples = 200

    #delimeter heirarchy:
    #    - > : > + 
    exp_filename = output_path*"align:$(A_root)+$(B_root)-beta:$β-max_iter:$max_iter-samples:$samples-seed:$seed-tol:$tol-results.json"


    relative_λ_diff, extremal_idx, eig_vals, extremal_vecs = SSHOPM_exp(A,B,samples,tol,max_iter,β,seed)

    #TODO: add in json save
    open(exp_filename,"w") do f 
        JSON.print(f,[relative_λ_diff, extremal_idx, eig_vals, extremal_vecs])
    end


end

function SSHOPM_exp(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor, samples::Int, tol::Float64,max_iter::Int,β::Float64,seed::UInt=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end 

    #A_V,A_Λ = SSHOPM_sample(A,samples,β,max_iter,tol)
    #B_V,B_Λ = SSHOPM_sample(B,samples,β,max_iter,tol)
    #AB_V,AB_Λ = SSHOPM_sample(A,B,samples,β,max_iter,tol)


    A_argmax_vec,A_argmax_val,A_Λ = SSHOPM_sample(A,samples,β,max_iter,tol)
    B_argmax_vec,B_argmax_val,B_Λ = SSHOPM_sample(B,samples,β,max_iter,tol)
    AB_argmax_vec,AB_argmax_val,AB_Λ = SSHOPM_sample(A,B,samples,β,max_iter,tol)

    A_i  = argmax([abs(x) for x in A_Λ])
    B_i  = argmax([abs(x) for x in B_Λ])
    AB_i = argmax([abs(x) for x in AB_Λ])

    relative_λ_diff = abs(A_argmax_val*B_argmax_val - AB_argmax_val)/abs(AB_argmax_val)

    println(relative_λ_diff)

    #r = rank(AB_V[:,:,AB_i])
   #U,S,Vt = svd(AB_V[:,:,AB_i])



    # not assuming anything about the rank of argmax(AB_Λ), 
    # error will trigger if not rank 1.
    #u_Avec_inner_prod = dot(U[:,1:r],A_V[:,A_i])
    #v_Bvec_inner_prod = dot(Vt[:,1:r],B_V[:,B_i])
    

    #  -- subspace angle  --  #
    #u_Av_subspaceAngle = acos(dot(U[:,1:r],A_V[:,A_i]))
    #v_Bv_subspaceAngle = acos(dot(Vt[:,1:r],B_V[:,B_i]))

    #  -- relative norm code  --  #
    #first_nnz_sign = x-> sign(x[findfirst(x .!= 0.0)])
    #relative_u_norm = norm( U[:,1:r]*Diagonal([first_nnz_sign( U[:,j]) for j in 1:r]) - first_nnz_sign(A_V[:,A_i])*A_V[:,A_i])/norm(A_V[:,A_i])
    #relative_v_norm = norm(Vt[:,1:r]*Diagonal([first_nnz_sign(Vt[:,j]) for j in 1:r]) - first_nnz_sign(B_V[:,B_i])*B_V[:,B_i])/norm(B_V[:,B_i])
   

    # -- group the variables to return from the experiment -- #
    #comparisons = (relative_λ_diff, u_Avec_inner_prod, v_Bvec_inner_prod)
    extremal_idx = (A_i, B_i, AB_i)
    eig_vals = (A_Λ, B_Λ, AB_Λ)
    extremal_vecs = (A_argmax_vec, B_argmax_vec, AB_argmax_vec)

    return relative_λ_diff, extremal_idx, eig_vals, extremal_vecs
end

function test_SSHOPM_exp_file(filename)

    relative_λ_diff, extremal_idx, eig_vals, extremal_vecs = open(filename,"r") do f 
        JSON.parse(f)
    end
    
    subspace_angle = dot(extremal_vecs[1]*extremal_vecs[2]',hcat(extremal_vecs[3]...))
    λ_A  = eig_vals[1][extremal_idx[1]]
    λ_B  = eig_vals[2][extremal_idx[2]]
    λ_AB = eig_vals[3][extremal_idx[3]]

    println("A.n = $(length(extremal_vecs[1]))  B.n=$(length(extremal_vecs[2]))")
    println("λ_A: $λ_A  -- λ_B: $λ_B  -- λ_AB: $λ_AB")
    println("λ_Aλ_B - λ_AB = $(λ_A*λ_B - λ_AB)")
    println("<(v_A*v_B, v_AB) = $subspace_angle")
end


"""-----------------------------------------------------------------------------
  Runs an instance of a graph alignment problem using a random graph model. Once 
  a graph is generated, we use the function 'ER_noise_model' to generate a 
  second network to align against it. Once the second network is generated it is 
  randomly permuted. All additional kwargs are passed to the 'align_matrices' 
  function.

  #TODO: update docs

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
function random_graph_exp(n::Int, perturbation_p::Float64,graph::RandomGraphType;
                          noise_model::NoiseModelFlag,seed=nothing,step_percent=.1,
                          use_metis=false,degreedist=nothing,p_edges=nothing,kwargs...)

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

    if typeof(noise_model) === ErdosRenyiNoise
        p_add = (p*perturbation_p)/(1-p)
        #B = ER_noise_model(A,0.0,perturbation_p)#,p_add)
        B = ER_noise_model(A,perturbation_p,p_add)
    elseif  typeof(noise_model) === DuplicationNoise
        steps = Int(ceil(step_percent*size(A,1))) #adds in 10% new nodes by default
        B,dup_vertices = duplication_perturbation_noise_model(A,steps, perturbation_p)
    end


    perm = shuffle(1:size(B,1))
    B = B[perm,perm]

    #return A, B , perm

	if use_metis
		apply_Metis_permutation!(A,100)
		apply_Metis_permutation!(B,100)
	end

    d_A = A*ones(size(A,1))
    d_B = B*ones(size(B,1))


    if typeof(noise_model) === DuplicationNoise
        perm,dup_vertices,align_matrices(A,B;kwargs...)
    else
        return d_A,d_B,perm,align_matrices(A,B;kwargs...)
    end
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

function duplication_perturbation_noise_model(A::SparseMatrixCSC{T,Int},steps::Integer, new_edge_p::Float64,seed::Union{UInt,Nothing}=nothing) where T
 
    is_undirected(A)                      || throw(ArgumentError("A must be undirected."))
    (new_edge_p >= 0 && new_edge_p <= 1)  || throw(ArgumentError("new_edge_p must be a probability."))
    steps >= 0                            || throw(ArgumentError("Must take a non-negative number of steps."))
    # let it steps equal 0 for testing purposes

    if seed !== nothing 
        Random.seed!(seed)
    end

    duplicated_vertices = Array{Int,1}(undef, steps)

    n,_ = size(A) # n will be updated

    #store A as an edge list so it's fast to sample
    A_edge_list = Array{Array{Tuple{Int,T},1},1}(undef,n+steps)
    for i = 1:n
        A_edge_list[i] = collect(zip(findnz(A[i,:])...))
    end
    for i = n+1:n+steps
        A_edge_list[i] = Array{Tuple{Int,T},1}(undef,0)
    end

    for step in 1:steps

        dup_vertex = rand(1:n)
        duplicated_vertices[step] = dup_vertex
        for (neighbor,weight) in A_edge_list[dup_vertex]
            if rand() < new_edge_p
                push!(A_edge_list[n+1],(neighbor,weight))
                push!(A_edge_list[neighbor],(n+1,weight))
            end
        end
        n += 1
    end



    #convert edge list back into a MatrixNetwork
    total_edges = 0
    for i=1:n
        total_edges += length(A_edge_list[i])
    end

    Is = Array{Int,1}(undef,total_edges)
    Js = Array{Int,1}(undef,total_edges)
    Vs = Array{T,1}(undef,total_edges)

    edge_idx = 1
    for i=1:n
        for (n_j,weight) in A_edge_list[i]
            Is[edge_idx] = i 
            Js[edge_idx] = n_j
            Vs[edge_idx] = weight
            edge_idx += 1
        end
    end

    return sparse(Js,Is,Vs,n,n), duplicated_vertices
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
function align_matrices(A::SparseMatrixCSC{T,Int},B::SparseMatrixCSC{S,Int};profile=false,
                        motif=Clique(),kwargs...) where {T,S}

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

        A_tensors = tensors_from_graph(A,kwargs[:orders],kwargs[:samples],motif)        
        B_tensors = tensors_from_graph(B,kwargs[:orders],kwargs[:samples],motif)
        
        A_motifCounts = [size(x.indices,2) for x in A_tensors]
        B_motifCounts = [size(x.indices,2) for x in B_tensors]

        subkwargs = Dict([(k,v) for (k,v) in kwargs if k != :orders && k != :samples])
        return A_motifCounts, B_motifCounts, align_tensors(A_tensors,B_tensors;subkwargs...)...

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
        triangle_count, gaped_triangles, _ = TAME_score(A_ten,B_ten,matching) 
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

