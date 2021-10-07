abstract type RandomGraphType end 
struct ErdosRenyi <: RandomGraphType end 
struct RandomGeometric <: RandomGraphType end 

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
