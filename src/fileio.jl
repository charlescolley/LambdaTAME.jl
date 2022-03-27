
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

		indices = Array{Int,2}(undef,order,m)
		values = Array{Float64,1}(undef,m)


		i = 1
		@inbounds for line in eachline(file)
			entries = split(chomp(line),'\t')

			indices[:,i] = [parse(Int,elem) for elem in entries[1:end-1]]
			if enforceFormatting
				sort!(indices[:,i])
			end
			values[i] = parse(Float64,entries[end])
			i += 1
		end

		#check for 0 indexing
		zero_indexed = false

		@inbounds for i in 1:m
		    if indices[1,i] == 0
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


function write_smat(A::SparseMatrixCSC{T,Int},path::String;delimeter::Char=',',kwargs...) where T
	@assert path[end-4:end] == ".smat"
	open(path,"w") do f 
		header = join([size(A)...,nnz(A)],delimeter)# ::NTuple{3,Int}
		println(f,header)
		for (i,j,v)=zip(findnz(A)...)
			i -= 1
			j -= 1
			print(f,i,delimeter,j,delimeter)
			println(f,v)
		end
	end
end