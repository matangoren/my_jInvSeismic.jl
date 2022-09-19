using CNNHelmholtzSolver
using KrylovMethods
using Statistics
using Flux
using Plots

# Eran Code
function absorbing_layer!(gamma::Array,pad,ABLamp;NeumannAtFirstDim=false)

    n=size(gamma)

    #FROM ERAN ABL:

    b_bwd1 = ((pad[1]:-1:1).^2)./pad[1]^2;
	b_bwd2 = ((pad[2]:-1:1).^2)./pad[2]^2;

	b_fwd1 = ((1:pad[1]).^2)./pad[1]^2;
	b_fwd2 = ((1:pad[2]).^2)./pad[2]^2;
	I1 = (n[1] - pad[1] + 1):n[1];
	I2 = (n[2] - pad[2] + 1):n[2];

	if NeumannAtFirstDim==false
		gamma[:,1:pad[2]] += ones(n[1],1)*b_bwd2'.*ABLamp;
		gamma[1:pad[1],1:pad[2]] -= b_bwd1*b_bwd2'.*ABLamp;
		gamma[I1,1:pad[2]] -= b_fwd1*b_bwd2'.*ABLamp;
	end

	gamma[:,I2] +=  (ones(n[1],1)*b_fwd2').*ABLamp;
	gamma[1:pad[1],:] += (b_bwd1*ones(1,n[2])).*ABLamp;
	gamma[I1,:] += (b_fwd1*ones(1,n[2])).*ABLamp;
	gamma[1:pad[1],I2] -= (b_bwd1*b_fwd2').*ABLamp;
	gamma[I1,I2] -= (b_fwd1*b_fwd2').*ABLamp;

    return gamma
end

fgmres_func = KrylovMethods.fgmres
c_type = ComplexF64
r_type = Float64

n=m=128
h=[1. /n, 1. /m]

blocks = 2
r_vcycle = zeros(c_type,n+1,m+1,1,1)
r_vcycle[floor(Int32,n / 2.0)-30,floor(Int32,m / 2.0),1,1] = r_type(1.0 ./mean(h.^2))
r_vcycle[floor(Int32,n / 2.0)+30,floor(Int32,m / 2.0),1,1] = r_type(1.0 ./mean(h.^2))
r_vcycle[floor(Int32,n / 2.0),floor(Int32,m / 2.0)-30,1,1] = r_type(1.0 ./mean(h.^2))
r_vcycle[floor(Int32,n / 2.0),floor(Int32,m / 2.0)+30,1,1] = r_type(1.0 ./mean(h.^2))
r_vcycle = vec(r_vcycle)
for i = 2:blocks
    r_vcycle1 = zeros(c_type,n+1,m+1,1,1)
    r_vcycle1[floor(Int32,(n / blocks)*(i-1)),floor(Int32,(m / blocks)*(i-1)),1,1] = r_type(1.0 ./mean(h.^2))
    global r_vcycle = cat(r_vcycle, vec(r_vcycle1), dims=2)
end

println("size of r_vcycle $(size(r_vcycle))")


# kappa = get2DSlowSquaredLinearModel(n,m)
kappa = r_type.(ones(n+1,m+1))
pad_cells = [10;10]
f = 10.0
gamma_val = 0.00001
omega = r_type(2*pi*f); # 2*pi*1.5 / (10*h[1])
gamma = gamma_val*2*pi * ones(r_type,size(kappa))
gamma = r_type.(absorbing_layer!(gamma, pad_cells, omega))


solver = getCnnHelmholtzSolver(;n=n,m=m,kappa=kappa, gamma=gamma, omega=omega)

result = solveLinearSystem(sparse(ones(size(r_vcycle))), r_vcycle, solver)

res1 = real(reshape(result[:,1],n+1, m+1))
println(size(res1))

heatmap(res1, color=:blues)
savefig("cnn_solver_point_source_result")