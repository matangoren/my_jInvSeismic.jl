# include("../../../../CNNHelmholtzSolver.jl/test/test_intro.jl")
# include("../../../../CNNHelmholtzSolver.jl/src/flux_components.jl")


using CNNHelmholtzSolver
using jInv.Mesh
using Helmholtz
using Flux
using SparseArrays
using CUDA
CUDA.allowscalar(true)
using Plots

println("NEW!!!")

r_type = Float32
function get_rhs(n, m, h; blocks=2)
    rhs = zeros(ComplexF64,n+1,m+1,1,1)
    rhs[floor(Int32,n / 2.0),floor(Int32,m / 2.0),1,1] = r_type(1.0 ./minimum(h)^2)
    rhs = vec(rhs)
    if blocks == 1
        return reshape(rhs, (length(rhs),1))
    end

    for i = 2:blocks
        rhs1 = zeros(ComplexF64,n+1,m+1,1,1)
        rhs1[floor(Int32,(n / blocks)*(i-1)),floor(Int32,(m / blocks)*(i-1)),1,1] = r_type(1.0 ./minimum(h)^2)
        rhs = cat(rhs, vec(rhs1), dims=2)
    end
    return rhs
end

n = 352
m = 240
domain = [0, 13.5, 0, 4.2]
h = r_type.([(domain[2]-domain[1])./ n, (domain[4]-domain[3])./ m])
n += 32
m += 16

M = getRegularMesh(domain,[n;m])
M.h = h
Ainv = CNNHelmholtzSolver.getCnnHelmholtzSolver("VU")
    
Helmholtz_param = HelmholtzParam(M,ones(5,5),ones(5,5),3.9*2*pi,true,true)
Ainv = CNNHelmholtzSolver.setMediumParameters(Ainv, Helmholtz_param)

rhs = get_rhs(M.n[1], M.n[2], M.h; blocks=8)
U, Ainv = CNNHelmholtzSolver.solveLinearSystem(sparse(ones(size(rhs))), rhs, Ainv)
heatmap(reshape(real(U[:,1])|>cpu,n+1,m+1), color=:blues)
savefig("package/e_solver")