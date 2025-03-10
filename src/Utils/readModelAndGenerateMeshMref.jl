
export readModelAndGenerateMeshMref
# 1.75 and 2.9 are the velocities that are suitable for the SEG salt model

function readModelAndGenerateMeshMref(readModelFolder::String,modelFilename::String,dim::Int64,pad::Int64,domain::Vector{Float64},newSize::Vector=[],velBottom::Float64=1.75,velHigh::Float64=2.9,
	doTranspose=true)
########################## m,mref are in Velocity here. ###################################

if dim==2
	# SEGmodel2Deasy.dat
	m = readdlm(string(readModelFolder,"/",modelFilename));
	m = m*1e-3;
	if doTranspose
		m = Matrix(m');
	else
		m = Matrix(m);
	end
	mref = getSimilarLinearModel(m,velBottom,velHigh);
else
	# 3D SEG slowness model
	file = matopen(string(readModelFolder,"/",modelFilename)); DICT = read(file); close(file);
	m = DICT["VELs"];
	m = m*1e-3;
	mref = getSimilarLinearModel(m,velBottom,velHigh);
end

sea = abs.(m[:] .- minimum(m)) .< 7e-2;
mref[sea] = m[sea];
if newSize!=[]
	m    = expandModelNearest(m,   collect(size(m)),newSize);
	mref = expandModelNearest(mref,collect(size(mref)),newSize);
end

Minv = getRegularMesh(domain,collect(size(m)));
println("In readModelAndGenerateMeshMref Minv- $(Minv.n) - $(Minv.domain) - $(Minv.h)")

(mPadded,MinvPadded) = addAbsorbingLayer(m,Minv,pad);
(mrefPadded,MinvPadded) = addAbsorbingLayer(mref,Minv,pad);
println("In readModelAndGenerateMeshMref MinvPadded- $(MinvPadded.n) - $(MinvPadded.domain)- $(MinvPadded.h)")


N = prod(MinvPadded.n);
boundsLow  = minimum(mPadded);
boundsHigh = maximum(mPadded);

boundsLow  = ones(N)*boundsLow;
boundsLow = convert(Array{Float32},boundsLow);
boundsHigh = ones(N)*boundsHigh;
boundsHigh = convert(Array{Float32},boundsHigh);

return (mPadded,MinvPadded,mrefPadded,boundsHigh,boundsLow);
end
