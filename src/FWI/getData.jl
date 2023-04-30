using CNNHelmholtzSolver
export getData

function get_rhs(n, m, h; blocks=2)
	r_type = Float32
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
function getData(m,pFor::FWIparam,doClear::Bool=false)
    # extract pointers
    M       	= pFor.Mesh
    omega   	= pFor.omega
	wavelet 	= pFor.WaveletCoef;
    gamma   	= pFor.gamma
    Q       	= pFor.Sources
    P       	= pFor.Receivers
	Ainv    	= pFor.ForwardSolver;
	batchSize 	= pFor.forwardSolveBatchSize;
	select  	= pFor.sourceSelection;

    nrec  		= size(P,2)
    nsrc  		= size(Q,2)

	An2cc = getNodalAverageMatrix(M; avN2C=avN2C_Nearest);

	println("########### In getData minimum m $(minimum(m)) size of m $(size(m))")
    m = An2cc'*m;
	gamma = An2cc'*gamma;
	println("########### In getData minimum m $(minimum(m)) size of m $(size(m))")

	println("### In getData ###")
	println(M.n)
	println(omega)

	# allocate space for data and fields
	n_nodes = prod(M.n.+1);
	# ALL AT ONCE DIRECT CODE
	H = GetHelmholtzOperator(M,m,omega, gamma, true,useSommerfeldBC);

	if isa(Ainv,ShiftedLaplacianMultigridSolver)
		Ainv.helmParam = HelmholtzParam(M,gamma,m,omega,true,useSommerfeldBC);
		H = H + GetHelmholtzShiftOP(m, omega,Ainv.shift[1]);
		H = sparse(H');
		# H is actually shifted laplacian now...
		Ainv.MG.relativeTol *= 1e-4;
	end

	
	if isa(Ainv, CnnHelmholtzSolver)
		Helmholtz_param = HelmholtzParam(M,gamma,m,omega,true,useSommerfeldBC)
		Ainv = setMediumParameters(Ainv, Helmholtz_param)
	end

	if select==[]
		Qs = Q*wavelet;
	else
		Qs = Q[:,select]*wavelet;
	end

	nsrc 		= size(Qs,2);

	if batchSize > nsrc
		batchSize = nsrc;
	end


	Fields = [];

	if doClear==false
		if pFor.useFilesForFields
			tfilename = getFieldsFileName(omega);
			tfile     = matopen(tfilename, "w");
		else
			Fields    = zeros(FieldsType,n_nodes   ,nsrc);
		end
	end

	numBatches 	= ceil(Int64,nsrc/batchSize);
	D 			= zeros(FieldsType,nrec,nsrc);
	U 			= zeros(FieldsType,n_nodes,batchSize);

	Ainv.doClear = 1;
	for k_batch = 1:numBatches
		println("handling batch ",k_batch," out of ",numBatches);
		batchIdxs = (k_batch-1)*batchSize + 1 : min(k_batch*batchSize,nsrc);
		if length(length(batchIdxs))==batchSize
			U[:] = convert(Array{FieldsType},Matrix(Qs[:,batchIdxs]));
		else
			U = convert(Array{FieldsType},Matrix(Qs[:,batchIdxs]));
		end
		U = get_rhs(M.n[1], M.n[2], M.h; blocks=length(batchIdxs)) # check point source
		println("In getData - before solveLinearSystem - H-$(size(H)) U-$(size(U)) batch-$(length(batchIdxs))")
		@time begin
			U,Ainv = solveLinearSystem(H,U,Ainv,0)
		end

		Ainv.doClear = 0;
		D[:,batchIdxs] = (P'*U);

		if doClear==false
			if pFor.useFilesForFields
				write(tfile,string("Ubatch_",k_batch),convert(Array{ComplexF64},U));
			else
				Fields[:,batchIdxs] = U;
			end
		end
	end

	if isa(Ainv,ShiftedLaplacianMultigridSolver)
		Ainv.MG.relativeTol *= 1e+4;
	end

	pFor.ForwardSolver = Ainv;

	if doClear==false
		if pFor.useFilesForFields
			close(tfile);
		else
			pFor.Fields = Fields;
		end
	end

	if !isa(Ainv, CnnHelmholtzSolver)
		if doClear
			clear!(pFor);
		elseif isa(Ainv,ShiftedLaplacianMultigridSolver)
			clear!(Ainv.MG); 
		end
	end
	# if doClear
	# 	clear!(pFor);
	# elseif isa(Ainv,ShiftedLaplacianMultigridSolver)
	# 	clear!(Ainv.MG); 
	# end
    return D,pFor
end
