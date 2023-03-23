using Revise
using Distributed
using DelimitedFiles
using MAT
using Multigrid.ParallelJuliaSolver
using jInvSeismic.FWI
using jInvSeismic.Utils
using Helmholtz
using Statistics
using jInv.InverseSolve
using jInv.LinearSolvers
using Multigrid

NumWorkers = 4;
if nworkers() == 1
	addprocs(NumWorkers);
elseif nworkers() < NumWorkers
 	addprocs(NumWorkers - nworkers());
end

@everywhere begin
	using jInv.InverseSolve
	using jInv.LinearSolvers
	using jInvSeismic.FWI
	using jInv.Mesh
	using Multigrid.ParallelJuliaSolver
	using jInv.Utils
	using DelimitedFiles
	using jInv.ForwardShare
	using KrylovMethods
	using CNNHelmholtzSolver
	using Flux
end

plotting = true;
if plotting
	using jInvVisPyPlot
	using PyPlot
	close("all")
end

@everywhere FWIDriversPath = "./";
include(string(FWIDriversPath,"prepareFWIDataFiles.jl"));
include(string(FWIDriversPath,"setupFWI.jl"));
# include("../../../src/FWI/adaptedMeshes.jl")

dataDir 	= pwd();
resultsDir 	= pwd();
modelDir 	= pwd();

########################################################################################################
windowSize = 4; # frequency continuation window size
simSrcDim = 16; # change to 1 for no simultaneous sources
maxBatchSize = 256; # use smaller value for 3D
useFilesForFields = false; # wheter to save fields to files


########## uncomment block for SEG ###############
 dim     = 2;
 pad     = 30;
 jumpSrc = 5;
 jumpRcv = 1;
 newSize = [600,300];

 (m,Minv,mref,boundsHigh,boundsLow) = readModelAndGenerateMeshMref(modelDir,
 	"examples/SEGmodel2Dsalt.dat",dim,pad,[0.0,13.5,0.0,4.2],newSize,1.752,2.9);

# omega = [3.0,3.3,3.6,3.9,4.2,4.5,5.0,5.5,6.5]*2*pi;
omega = [2.5,3.0,3.3,3.6,3.9]*2*pi;
offset  = newSize[1];
println("Offset is: ",offset," cells.")
alpha1 = 5e0;
alpha2 = 5e1;
stepReg = 5e1;
EScycles = 2;
cgit = 7;

freqContSweeps = 5;
freqRanges = [(1,4), (1,4), (4,length(omega)), (4,length(omega)),
		(length(omega), length(omega))];
regularizations = ["high", "high", "low", "low", "low"];
GNiters = [50, 50, 15 ,15, 100];

# ###################################################################################################################
dataFilenamePrefix = string(dataDir,"/DATA_",tuple((Minv.n)...));
resultsFilename = string(resultsDir,"/FWI_",tuple((Minv.n)...));
#######################################################################################################################
writedlm(string(resultsFilename,"_mtrue.dat"),convert(Array{Float16},m));
writedlm(string(resultsFilename,"_mref.dat"),convert(Array{Float16},mref));
resultsFilename = string(resultsFilename,".dat");

println("omega*maximum(h): ",omega*maximum(Minv.h)*sqrt(maximum(1.0./(boundsLow.^2))));
ABLpad = pad + 4;
# Ainv  = getParallelJuliaSolver(ComplexF64,Int64,numCores=16,backend=1);
# Ainv = getJuliaSolver();
Ainv = getCnnHelmholtzSolver() # need to pass parameters - 

workersFWI = workers();
println(string("The workers that we allocate for FWI are:",workersFWI));

figure(1,figsize = (22,10));
plotModel(m,includeMeshInfo=true,M_regular = Minv,cutPad=pad,limits=[1.5,4.5],figTitle="mref",filename="orig.png");

figure(2,figsize = (22,10));
plotModel(mref,includeMeshInfo=true,M_regular = Minv,cutPad=pad,limits=[1.5,4.5],figTitle="mref",filename="mref.png");


prepareFWIDataFiles(m,Minv,mref,boundsHigh,boundsLow,dataFilenamePrefix,omega,ones(ComplexF64,size(omega)),
									pad,ABLpad,jumpSrc,jumpRcv,offset,workersFWI,maxBatchSize,Ainv,useFilesForFields);