export getMeshAdaptedParams

function getMeshAdaptedParams(Minv::RegularMesh,omega::Array{Float64},Q::SparseMatrixCSC,P::SparseMatrixCSC,gamma::Array{Float64})
nfreqs = length(omega);
omega_max = maximum(omega);
Qs = Array{SparseMatrixCSC}(undef,nfreqs);
Ps = Array{SparseMatrixCSC}(undef,nfreqs);
gammas = Array{Vector{Float64}}(undef,nfreqs);
Mfwds = Array{RegularMesh}(undef,nfreqs);

MakeSureDividesIn = 16; # needed for the CNNHelmholtzSolver
roundToNearestDivisable = (x::Array) -> (floor.(Int64,(x.-1)/MakeSureDividesIn.+1)*MakeSureDividesIn)

for k = 1:nfreqs
	omega_k = omega[k];
	ratio = omega_k/omega_max;
	if ratio < 0.95
		Minv_nodal = getRegularMesh(Minv.domain,Minv.n.+1);
		# roundToNearestDivisable
		Mfwd = getRegularMesh(Minv.domain,roundToNearestDivisable(Minv.n.*ratio));
		Mfwd_nodal = getRegularMesh(Minv.domain,Mfwd.n.+1);
		Interp = prepareMesh2Mesh(Mfwd, Minv,false);
		Interp_nodal = prepareMesh2Mesh(Mfwd_nodal, Minv_nodal,false);
		gammas[k] = interpGlobalToLocal(vec(gamma),Interp);
		Qs[k]     = interpGlobalToLocal(Q,Interp_nodal);
		Ps[k]     = interpGlobalToLocal(P,Interp_nodal);
		println("************************** Mesh2Mesh size of Interp: ",size(Interp))
		Mfwds[k] = Mfwd;
	# if ratio < 0.75
		# Mfwd = Minv
		# Minv_nodal = getRegularMesh(Minv.domain,Minv.n.+1);
		# Mfwd_nodal = Minv_nodal;
		# if ratio < 0.5
			# println("************************** Mesh2Mesh 1/2 *******************************")
			# Mfwd = getRegularMesh(Minv.domain,div.(Minv.n,2));
			# Mfwd_nodal = getRegularMesh(Minv.domain,Mfwd.n.+1);
			# println(Minv.n)
			# println(Mfwd.n)
		# else
			# println("************************** Mesh2Mesh 3/4 *******************************")
			# Mfwd = getRegularMesh(Minv.domain,div.(3*Minv.n,4));
			# Mfwd_nodal = getRegularMesh(Minv.domain,Mfwd.n.+1);
			# println(Minv.n)
			# println(Mfwd.n)
		# end
		
		# Interp = prepareMesh2Mesh(Mfwd, Minv,false);
		# Interp_nodal = prepareMesh2Mesh(Mfwd_nodal, Minv_nodal,false);
		# gammas[k] = interpGlobalToLocal(vec(gamma),Interp);
		# Qs[k] = interpGlobalToLocal(Q,Interp_nodal);
		# Ps[k] = interpGlobalToLocal(P,Interp_nodal);
		# Mfwds[k] = Mfwd;
	else
		println("************************** Mesh2Mesh no change *******************************")
		Qs[k] = Q;
		Ps[k] = P;
		gammas[k] = vec(gamma);
		Mfwds[k] = Minv;
	end
end
return (Mfwds,gammas,Qs,Ps)
end
