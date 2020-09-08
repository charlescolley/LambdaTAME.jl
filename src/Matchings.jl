
function low_rank_matching(U::Array{Float64,2},V::Array{Float64,2})
    n,d1 = size(U)
    m,d2 = size(V)
    @assert d1 == d2

    matching_weights = zeros(d1)
    matchings = Array{Dict{Int,Int},1}(undef,d1)

    for i = 1:d1
        matchings[i],matching_weights[i] = rank_one_matching(U[:,i],V[:,i])
    end


    D = zeros(d1,d2)
    for j in 1:d2
        for i in 1:d1
            w = 0.0
            for (v_i,v_j) in matchings[j]
                edge_w = U[v_i,i]*V[v_j,i]
                if edge_w > 0
                    w += edge_w
                end
            end
            D[i,j] = matching_weights[i]/w
        end
    end

    d_j = maximum(D,dims= 1)
    opt_j = argmin(d_j).I[2]

    return matchings[opt_j], matching_weights[opt_j]

end


function rank_one_matching(u::Array{Float64,1},v::Array{Float64,1})

    #get maximum matching by rearrangement theorem
    u_perm = sortperm(u,rev=true)
    v_perm = sortperm(v,rev=true)

    matching_weight = 0
    Match_mapping = Dict{Int,Int}()
    for (i,j) in zip(u_perm,v_perm)

        #only match positive to positive and negative to negative
        if (u[i] > 0 && v[j] > 0 ) || (u[i] < 0 && v[j] < 0)
            Match_mapping[i] = j
            matching_weight += u[i]*v[j]
        end
    end
    return Match_mapping, matching_weight

end

function search_Krylov_space(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,U::Array{Float64,2},V::Array{Float64,2})

    best_score = -1
    best_i = -1
    best_j = -1

    Triangle_check = Dict{Array{Int,1},Int}()
    A_unique_nnz = length(A.values)
    B_unique_nnz = length(B.values)

    if A_unique_nnz > B_unique_nnz
        for i in 1:A_unique_nnz
            Triangle_check[A.indices[i,:]] = 1
        end
        Input_tensor = B
    else
        for i in 1:B_unique_nnz
            Triangle_check[B.indices[i,:]] = 1
        end
        Input_tensor = A
    end

    for i in 1:size(U,2)
       for j in 1:size(V,2)

            if A_unique_nnz > B_unique_nnz
                matched_tris,gaped_tris = TAME_score(Triangle_check,Input_tensor,V[:,j],U[:,i])
            else
                matched_tris,gaped_tris = TAME_score(Triangle_check,Input_tensor,U[:,i],V[:,i])
            end

            if matched_tris > best_score
                best_score = matched_tris
                best_i = i
                best_j = j
            end
        end
    end
    return best_score, best_i, best_j
end

function search_Krylov_space(A::COOTen,B::COOTen,U::Array{Float64,2},V::Array{Float64,2})

    best_score = -1
    best_i = -1
    best_j = -1

    Triangle_check = Dict{Array{Int,1},Int}()

    if A.unique_nnz > B.unique_nnz
        for i in 1:A.unique_nnz
            Triangle_check[A.indices[i,:]] = 1
        end
        Input_tensor = B
    else
        for i in 1:B.unique_nnz
            Triangle_check[B.indices[i,:]] = 1
        end
        Input_tensor = A
    end

    for i in 1:size(U,2)
       for j in 1:size(V,2)

            if A.unique_nnz > B.unique_nnz
                matched_tris,gaped_tris = TAME_score(Triangle_check,Input_tensor,V[:,j],U[:,i])
            else
                matched_tris,gaped_tris = TAME_score(Triangle_check,Input_tensor,U[:,i],V[:,i])
            end

            if matched_tris > best_score
                best_score = matched_tris
                best_i = i
                best_j = j
            end
        end
    end
    return best_score, best_i, best_j
end

#used when we don't want to recreate the triangle matching dictionary multiple times
function TAME_score(Triangle_Dict::Dict{Array{Int,1},Int},Input_tensor,
                    u::Array{Float64,1},v::Array{Float64,1})

    Match_mapping, _ = rank_one_matching(u,v)
    TAME_score(Triangle_Dict,Input_tensor,Match_mapping)

end


#Computes the TAME score for this iterate by
function TAME_score(A::COOTen,B::COOTen,u::Array{Float64,1},v::Array{Float64,1})

    Match_mapping, weight = rank_one_matching(u,v)
    TAME_score(A,B,Match_mapping)

end

function TAME_score(A::COOTen,B::COOTen,U::Array{Float64,2},V::Array{Float64,2})

    Match_mapping = low_rank_matching(U,V)
    TAME_score(A,B,Match_mapping)

end

function TAME_score(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,U::Array{Float64,2},V::Array{Float64,2};return_timings=false)

    if return_timings
        (Match_mapping, _), matching_time = @timed low_rank_matching(U,V)
        (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,Match_mapping)
        return triangle_count, gaped_triangles, matching_time, scoring_time 
    else
        Match_mapping,weight = low_rank_matching(U,V)
        TAME_score(A,B,Match_mapping)
    end

end

function TAME_score(A::ThirdOrderSymTensor, B::ThirdOrderSymTensor, X::SparseMatrixCSC{Float64,Int64};return_timings=false)

    if return_timings
        x ,bipartite_matching_time = @timed bipartite_matching(X)
        (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,Dict(i => j for (i,j) in enumerate(x.match)))
        return triangle_count, gaped_triangles, bipartite_matching_time, scoring_time
    else
        x = bipartite_matching(X)
        return TAME_score(A,B,Dict(i => j for (i,j) in enumerate(x.match)))
    end

end

function TAME_score(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,X::Array{Float64,2};return_timings=false)

    if return_timings
        (_,_,matching,_) ,matching_time = @timed bipartite_matching_primal_dual(X)
        (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
        return triangle_count, gaped_triangles, matching_time, scoring_time, matching
    else
        _,_,matching,_ = bipartite_matching_primal_dual(X)
        return TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
    end
end

function TAME_score(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,x::Array{Float64,1};return_timings=false)

    X = reshape(x,A.n,B.n)
    if return_timings
        (_,_,matching,_) ,hungarian_time = @timed bipartite_matching_primal_dual(X)
        (triangle_count, gaped_triangles), matching_time = @timed TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
        return triangle_count, gaped_triangles, hungarian_time, matching_time
    else
        _,_,matching,_ = bipartite_matching_primal_dual(X)
        return TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
    end

end


function TAME_score(A::COOTen,B::COOTen,X::Array{Float64,2};return_timings=false)

   if return_timings
        (_,_,matching,_) ,scoring_time = @timed bipartite_matching_primal_dual(X)
        (triangle_count, gaped_triangles), matching_time = @timed TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
        return triangle_count, gaped_triangles, matching_time, matching_time
    else
        (_,_,matching,_) ,scoring_time = @timed bipartite_matching_primal_dual(X)
        return TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
    end

end

function TAME_score(A::COOTen,B::COOTen,x::Array{Float64,1};return_timings=false)

   if return_timings
        (_,_,matching) ,matching_time = @timed bipartite_matching_primal_dual(reshape(x,A.cubical_dimension,B.cubical_dimension))
        (triangle_count, gaped_triangles), scoring_time = @timed TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
        return triangle_count, gaped_triangles, matching_time,scoring_time
    else
        _,_,matching = bipartite_matching_primal_dual(reshape(x,A.cubical_dimension,B.cubical_dimension))
        return TAME_score(A,B,Dict(j => i for (i,j) in enumerate(matching)))
    end

end

function TAME_score(A::ThirdOrderSymTensor,B::ThirdOrderSymTensor,Match_mapping::Dict{Int,Int})

    match_len = length(Match_mapping)

    Triangle_check = Dict{Array{Int,1},Int}()
    gaped_triangles = 0
    triangle_count = 0

    if size(A.indices,1) > size(B.indices,1)

        for i in 1:size(A.indices,1)
            Triangle_check[A.indices[i,:]] = 1
        end

        #invert to map v indices to u
        Match_mapping = Dict(value => key for (key, value) in Match_mapping)

        for i in 1:size(B.indices,1)
            v_i,v_j,v_k = B.indices[i,:]

            matched_triangle =
              sort([get(Match_mapping,v_i,-1),get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])
    #        println(B.indices[i,:]," -> ",matched_triangle)
            match = get(Triangle_check,matched_triangle,0)
            if match == 1
                triangle_count += 1
            else
                gaped_triangles += 1
            end
        end

    else
        for i in 1:size(B.indices,1)
            Triangle_check[B.indices[i,:]] = 1
        end

        for i in 1:size(A.indices,1)
            v_i,v_j,v_k = A.indices[i,:]
            matched_triangle =
               sort([get(Match_mapping,v_i,-1), get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])

            match = get(Triangle_check,matched_triangle,0)
            if match == 1
                triangle_count += 1
            else
                gaped_triangles += 1
            end
        end
    end

   # triangles, triangle_weight = count_triangles(sub_A,sub_B)

    return triangle_count, gaped_triangles #sub_A, sub_B

end

function TAME_score(A::COOTen,B::COOTen,Match_mapping::Dict{Int,Int})

    match_len = length(Match_mapping)

    Triangle_check = Dict{Array{Int,1},Int}()
    gaped_triangles = 0
    triangle_count = 0

    if A.unique_nnz > B.unique_nnz

        for i in 1:A.unique_nnz
            Triangle_check[A.indices[i,:]] = 1
        end

        #invert to map v indices to u
        Match_mapping = Dict(value => key for (key, value) in Match_mapping)

        for i in 1:B.unique_nnz
            v_i,v_j,v_k = B.indices[i,:]

            matched_triangle =
              sort([get(Match_mapping,v_i,-1),get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])
    #        println(B.indices[i,:]," -> ",matched_triangle)
            match = get(Triangle_check,matched_triangle,0)
            if match == 1
                triangle_count += 1
            else
                gaped_triangles += 1
            end
        end

    else
        for i in 1:B.unique_nnz
            Triangle_check[B.indices[i,:]] = 1
        end

        for i in 1:A.unique_nnz
            v_i,v_j,v_k = A.indices[i,:]
            matched_triangle =
               sort([get(Match_mapping,v_i,-1), get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])

            match = get(Triangle_check,matched_triangle,0)
            if match == 1
                triangle_count += 1
            else
                gaped_triangles += 1
            end
        end
    end

   # triangles, triangle_weight = count_triangles(sub_A,sub_B)

    return triangle_count, gaped_triangles #sub_A, sub_B

end

function TAME_score(Triangle_Dict::Dict{Array{Int,1},Int},Input_tensor::COOTen,
                    Match_mapping::Dict{Int,Int})

    triangle_count = 0
    gaped_triangles = 0

    for i in 1:Input_tensor.unique_nnz
        v_i,v_j,v_k = Input_tensor.indices[i,:]

        matched_triangle =
          sort([get(Match_mapping,v_i,-1),get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])

        match = get(Triangle_Dict,matched_triangle,0)
        if match == 1
            triangle_count += 1
        else
            gaped_triangles += 1
        end
    end
    return triangle_count, gaped_triangles
end

function TAME_score(Triangle_Dict::Dict{Array{Int,1},Int},Input_tensor::ThirdOrderSymTensor,
                    Match_mapping::Dict{Int,Int})

    triangle_count = 0
    gaped_triangles = 0

    for i in 1:length(Input_tensor.values)
        v_i,v_j,v_k = Input_tensor.indices[i,:]

        matched_triangle =
          sort([get(Match_mapping,v_i,-1),get(Match_mapping,v_j,-1),get(Match_mapping,v_k,-1)])

        match = get(Triangle_Dict,matched_triangle,0)
        if match == 1
            triangle_count += 1
        else
            gaped_triangles += 1
        end
    end
    return triangle_count, gaped_triangles
end




function bipartite_matching_primal_dual(X::Matrix{Float64};tol::Float64=1e-8,
                                        normalize_weights::Bool=false)
    #to get the access pattern right, we must match the right hand side to the left hand side. 

	m,n = size(X)
	@assert m >= n  #error occurs when m < n 

    # variables used for the primal-dual algorithm
    # normalize ai values # updated on 2-19-2019
    if normalize_weights
        X ./= maximum(abs.(X))
    end

	#initialize variables
    alpha=zeros(Float64,n)
    bt=zeros(Float64,m+n)#beta
    match1 = zeros(Int64,n)
    match2 = zeros(Int64,n+m)
    queue=zeros(Int64,n)
    t=zeros(Int64,m+n)
    tmod = zeros(Int64,m+n)
    ntmod=0

    # initialize the primal and dual variables

	for j = 1:n
		for i=1:m
			if X[i,j] > alpha[j]
			   alpha[j]=X[i,j]
			end
		end
    end


    # dual variables (bt) are initialized to 0 already
    # match1 and match2 are both 0, which indicates no matches

    j=1
    @inbounds while j<=n

        for i=1:ntmod
            t[tmod[i]]=0
        end
        ntmod=0
        # add i to the stack
        head=1
        tail=1
        queue[head]=j
        while head <= tail && match1[j]==0 #queue empty + i is unmatched
            
 		 	k=queue[head]
			#println("begining of queue loop")

            for i=1:m+1 #iterate over column k

				if i == m+1 #check the dummy node
					i = k+m
				end
				if i == k+m #dummy nodes don't have weight
					if 0.0 < alpha[k] + bt[i] - tol
						continue
					end
                elseif X[i,k] < alpha[k] + bt[i] - tol
					continue
                end # skip if tight

                if t[i]==0
                    tail=tail+1 #put the potential match in the queue
                    if tail <= m
                        queue[tail]=match2[i]
					end
                    t[i]=k  #try vertex k for vertex j
                    ntmod=ntmod+1
                    tmod[ntmod]=i
                    if match2[i]<1  #if i is unmatched
                        while i>0   #unfurl out to get an augmented path
                            match2[i]=t[i]
                            k=t[i]
                            temp=match1[k]
                            match1[k]=i
							i=temp
						end
                        break
                    end
                end
            end
            head=head+1

        end

        #if node j was unable to be matched, update flows and search for new augmenting path
		if match1[j] < 1
            theta=Inf
            for i=1:head-1
                t1=queue[i]
                for t2=1:m
                    if t[t2] == 0 && alpha[t1] + bt[t2] - X[t2,t1] < theta
                        theta = alpha[t1] + bt[t2] - X[t2,t1]
                    end
                end
				#check t1's dummy node
				if t[t1 + m] == 0 && alpha[t1] + bt[t1 + m] < theta
                        theta = alpha[t1] + bt[t1 + m]
				end
            end

            for i=1:head-1
                alpha[queue[i]] -= theta
            end
            for i=1:ntmod
                bt[tmod[i]] += theta
            end
            continue
        end

        j=j+1
    end

	#count
    val=0.0
    for j=1:n
        for i=1:m
            if i==match1[j]
                val=val+X[i,j]
            end
        end
    end

	#count how many are properly matched
    noute = 0
    for j=1:n
        if match1[j]<=m
            noute=noute+1
        end
	end
	
    return val,noute,match1, match2
end