#=------------------------------------------------------------------------------
                        Formatting Routines
------------------------------------------------------------------------------=#
function produce_ssten_from_triangles(file)

    A = MatrixNetworks.readSMAT(file)
    (n,m) = size(A)
    if(n != m)
        println("rectangular")
    end

	if !issymmetric(A)
		A = max.(A,A')  #symmetrize for Triangles routine
	end
	print(issymmetric(A))

    T = collect(MatrixNetworks.triangles(A))

    output_file = alterfilename(file,".ssten",".",false)

    open(output_file,"w") do f
        write(f,"$(3)\t$(n)\t$(length(T))\n")

        for (v_i,v_j,v_k) in T
            write(f,"$(v_i)\t$(v_j)\t$(v_k)\t1.0\n")
        end
    end

end

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
                       ThirdOrderSparse=true,kwargs...)

    if ThirdOrderSparse
        A = load_ThirdOrderSymTensor(graph_A_file)
        B = load_ThirdOrderSymTensor(graph_B_file)
    else

        A = load(graph_A_file,false,"COOTen")
        B = load(graph_B_file,false,"COOTen")
    end

	return align_tensors(A,B;kwargs...)

end

function get_TAME_ranks(graph_A_file::String,graph_B_file::String)

    A = load_ThirdOrderSymTensor(graph_A_file)
    B = load_ThirdOrderSymTensor(graph_B_file)

    results = Dict()

    max_iter = 30
    tol = 1e-12
    X_0 = ones(A.n,B.n)
    X_0 ./=norm(X_0)
    #V_0 = ones(B.n,1)

   # U_0 ./= norm(U_0)
    #V_0 ./= norm(V_0)

    alphas = [1.0,.5]
    betas =[0.0,1.0,10]

    for α in alphas
        for β in betas
            experiment_string = "α:$(α)_β:$(β)"
            #_,ranks = lowest_rank_TAME_original_svd_for_ranks(A,B,U_0,V_0,β, max_iter,tol,α)
            _,_,profile = TAME(A,B,X_0,β, max_iter,tol,α;profile=true)
            results[experiment_string] = profile
            println("finished $(split(graph_A_file,"/")[end])--$(split(graph_B_file,"/")[end]):$experiment_string")
        end
    end

    return results
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


function distributed_pairwise_alignment(dir::String;kwargs...)

    #align all .ssten files
    ssten_files = sort([f for f in readdir(dir) if occursin(".ssten",f)])
    distributed_pairwise_alignment(ssten_files,dir;kwargs...)

end

function distributed_pairwise_alignment(files::Array{String,1},dirpath::String;
                                        method="LambdaTAME",kwargs...)

    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")

    futures = []
	exp_results = []

#    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files),length(ssten_files))

    for i in 1:length(files)
        for j in i+1:length(files)

            #TODO: make this more robust
            future = @spawn align_tensors(dirpath*files[i],dirpath*files[j];method=method,kwargs...)
            push!(futures,((i,j),future))
        end
    end

    for ((i,j), future) in futures

		if method == "LambdaTAME" ||  method == "LowRankTAME"
			matched_tris, max_tris, _, _, results = fetch(future)
		elseif method == "TAME"
			matched_tris, max_tris, _, results = fetch(future)
		end
		push!(exp_results,(files[i],files[j],matched_tris, max_tris, results))

    end

    return files, exp_results
end


function distributed_TAME_rank_experiment(files::Array{String,1},dirpath::String;kwargs...)

    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")


    futures = []
    exp_results = []

#    Best_alignment_ratio = Array{Float64}(undef,length(ssten_files),length(ssten_files))

    for i in 1:length(files)
        for j in i+1:length(files)

            #TODO: make this more robust
            future = @spawn get_TAME_ranks(dirpath*files[i],dirpath*files[j];kwargs...)
            push!(futures,((i,j),future))
        end
    end

    for ((i,j), future) in futures
        push!(exp_results,(files[i],files[j],fetch(future)))
    end

    return files, exp_results
end


function self_alignment(dir::String;kwargs...)

    #align all .ssten files
    ssten_files = sort([dir*"/"*f for f in readdir(dir) if occursin(".ssten",f)])
    return self_alignment(ssten_files;kwargs...)

end

#=------------------------------------------------------------------------------
                      Generated Graph Experiments Routines
------------------------------------------------------------------------------=#

function distributed_random_trials(trial_count::Int,process_count::Int,seed_exps::Bool=false
                                   ;method="LambdaTAME",graph_type::String="ER",
								   n_sizes = [10, 50, 100, 500, 1000, 2000,5000],
								   kwargs...)

    #only handling even batch sizes
    @assert trial_count % process_count == 0

    #ensure file is loaded on all processes
    @everywhere include_string(Main,$(read("LambdaTAME.jl",String)),"LambdaTAME.jl")

    p_remove = [.01,.05]
	batches = Int(trial_count/process_count)

	if seed_exps
		Random.seed!(0)
		seeds = rand(UInt64, length(p_remove),length(n_sizes), trial_count)
	end
	seed = nothing
	results = []
    p_index = 1

    for p in p_remove

        n_index = 1

        for n in n_sizes


            for batch in 1:batches

                futures = []

                for i in 1:process_count

					if seed_exps
						seed = seeds[p_index,n_index,(batch-1)*batches + i]
					end

					future = @spawn random_graph_exp(n,p,graph_type;seed=seed,method=method,kwargs...)
                    push!(futures,((batch-1)*batches + i,seed,p,n,future))

                end

                for (i,seed,p,n,future) in futures

					if method == "LambdaTAME" ||  method == "LowRankTAME"
						matched_tris, max_tris, _, _, exp_results = fetch(future)
					elseif method == "TAME"
    					matched_tris, max_tris, _, exp_results = fetch(future)
					end
					push!(results,(i, seed, p, n, matched_tris, max_tris, exp_results))
                end
            end

            n_index += 1
        end
        p_index += 1
    end

    return results

end
#,graph_type::String="ER"
function random_graph_exp(n::Int, p_remove::Float64,graph_type::String;seed=nothing,degreedist=nothing,kwargs...)

	if seed !== nothing
		Random.seed!(seed)
	end

	if graph_type == "ER"
		 p = 2*log(n)/n
		 A = erdos_renyi(n,p)

	elseif graph_type == "RandomGeometric"

		if degreedist === nothing
			k = 10
			p = k/n
			A = random_geometric_graph(n,k)
		else
			d = 2
			A = spatial_network(n, d;degreedist= degreedist)
			p = nnz(A)/n^2
		end
	elseif graph_type == "HyperKron"
		p = .4#2*log(n)/n
		r = .4

		A = sparse(gpa_graph(n,p,r,5))

	else
		error("invalid graph type: $graph_type\n must be either 'ER','RandomGeometric' or 'HyperKron'")
	end

	p_add = p*p_remove/(1-p)
	B = ER_noise_model(A,n,p_remove,p_add)
	return set_up_tensor_alignment(A,B;kwargs...)
end

function set_up_tensor_alignment(A,B;kwargs...)

	n,n = size(A)
	#permute the rows of B
    perm = shuffle(1:n)
    B = B[perm,perm]

    #build COOTens from graphs
    A_tris = collect(MatrixNetworks.triangles(A))
    B_tris = collect(MatrixNetworks.triangles(B))

    A_nnz = length(A_tris)
    B_nnz = length(B_tris)

    A_indices = Array{Int,2}(undef,A_nnz,3)
    B_indices = Array{Int,2}(undef,B_nnz,3)
    for i =1:A_nnz
        A_indices[i,1]= A_tris[i][1]
        A_indices[i,2]= A_tris[i][2]
        A_indices[i,3]= A_tris[i][3]
    end

    for i =1:B_nnz
        B_indices[i,1]= B_tris[i][1]
        B_indices[i,2]= B_tris[i][2]
        B_indices[i,3]= B_tris[i][3]
    end

    A_vals = ones(A_nnz)
    B_vals = ones(B_nnz)

    A_ten = ThirdOrderSymTensor(n,A_indices,A_vals)
    B_ten = ThirdOrderSymTensor(n,B_indices,B_vals)
    return align_tensors(A_ten,B_ten;kwargs...)
end


function ER_noise_model(A,n::Int,p_remove::F,p_add::F) where {F <: AbstractFloat}
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

    return B
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

function spatial_network(n::Integer, d::Integer; degreedist=LogNormal(log(4),1))
  xy, ei, ej = spatial_graph_edges(n, d;degreedist=degreedist)
  A = sparse(ei,ej,1.0,n,n)
  return max.(A,A')
end

function spatial_graph_edges(n::Integer,d::Integer;degreedist=LogNormal(log(4),1))
  xy = rand(d,n)
  T = BallTree(xy)
  # form the edges for sparse
  ei = Int[]
  ej = Int[]
  for i=1:n
    deg = ceil(Int,minimum(rand(degreedist),n))
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

