#include "CUDAImageSolver.h"
#include "../../shared/OptUtils.h"
#include "ConvergenceAnalysis.h"

extern "C" double solveSFSStub(SolverInput& input, SolverState& state, SolverParameters& parameters, ConvergenceAnalysis<float>* ca);
extern "C" void solveSFSEvalCurrentCostJTFPreAndJTJStub(SolverInput& input, SolverState& state, SolverParameters& parameters, float* costResult, float* jtfResult, float* preResult, float* jtjResult);


CUDAImageSolver::CUDAImageSolver(const std::vector<unsigned int>& dims) : m_dims(dims)
{
	const unsigned int THREADS_PER_BLOCK = 1024; // keep consistent with the GPU
	const unsigned int tmpBufferSize = THREADS_PER_BLOCK*THREADS_PER_BLOCK;

    const unsigned int N = dims[0] * dims[1];
    const size_t unknownStorageSize = sizeof(float)*N;

	// State
	cudaSafeCall(cudaMalloc(&m_solverState.d_delta,		unknownStorageSize));
    cudaSafeCall(cudaMalloc(&m_solverState.d_r,            unknownStorageSize));
    cudaSafeCall(cudaMalloc(&m_solverState.d_z,            unknownStorageSize));
    cudaSafeCall(cudaMalloc(&m_solverState.d_p,            unknownStorageSize));
    cudaSafeCall(cudaMalloc(&m_solverState.d_Ap_X,         unknownStorageSize));
	cudaSafeCall(cudaMalloc(&m_solverState.d_scanAlpha,	sizeof(float)));
	cudaSafeCall(cudaMalloc(&m_solverState.d_scanBeta,		sizeof(float)));
	cudaSafeCall(cudaMalloc(&m_solverState.d_rDotzOld,		sizeof(float)*N));
    cudaSafeCall(cudaMalloc(&m_solverState.d_preconditioner, unknownStorageSize));
	cudaSafeCall(cudaMalloc(&m_solverState.d_sumResidual,	sizeof(float)));

    // Solver-specific intermediates
    cudaSafeCall(cudaMalloc(&m_solverState.B_I    , sizeof(float)*N));
    cudaSafeCall(cudaMalloc(&m_solverState.B_I_dx0, sizeof(float)*N));
    cudaSafeCall(cudaMalloc(&m_solverState.B_I_dx1, sizeof(float)*N));
    cudaSafeCall(cudaMalloc(&m_solverState.B_I_dx2, sizeof(float)*N));
    cudaSafeCall(cudaMalloc(&m_solverState.pguard, sizeof(bool)*N));
}

CUDAImageSolver::~CUDAImageSolver()
{
	// State
	cudaSafeCall(cudaFree(m_solverState.d_delta));
	cudaSafeCall(cudaFree(m_solverState.d_r));
	cudaSafeCall(cudaFree(m_solverState.d_z));
	cudaSafeCall(cudaFree(m_solverState.d_p));
	cudaSafeCall(cudaFree(m_solverState.d_Ap_X));
	cudaSafeCall(cudaFree(m_solverState.d_scanAlpha));
	cudaSafeCall(cudaFree(m_solverState.d_scanBeta));
	cudaSafeCall(cudaFree(m_solverState.d_rDotzOld));
    cudaSafeCall(cudaFree(m_solverState.d_preconditioner));
	cudaSafeCall(cudaFree(m_solverState.d_sumResidual));

    // Solver-specific intermediates
    cudaSafeCall(cudaFree(m_solverState.B_I    ));
    cudaSafeCall(cudaFree(m_solverState.B_I_dx0));
    cudaSafeCall(cudaFree(m_solverState.B_I_dx1));
    cudaSafeCall(cudaFree(m_solverState.B_I_dx2));
}

double CUDAImageSolver::solve(const NamedParameters& solverParams, const NamedParameters& probParams, bool profileSolve, std::vector<SolverIteration>& iters)
{

    m_solverState.d_x = getTypedParameterImage<float>("X", probParams);

    SolverInput solverInput;
    solverInput.N = m_dims[0] * m_dims[1];
    solverInput.width = m_dims[0];
    solverInput.height = m_dims[0];

    solverInput.d_targetIntensity = getTypedParameterImage<float>("Im", probParams);
    solverInput.d_targetDepth = getTypedParameterImage<float>("D_i", probParams);
    solverInput.d_depthMapRefinedLastFrameFloat = nullptr;
    solverInput.d_maskEdgeMapR = getTypedParameterImage<unsigned char>("edgeMaskR", probParams);
    solverInput.d_maskEdgeMapC = getTypedParameterImage<unsigned char>("edgeMaskC", probParams);
    
    NamedParameters::Parameter lightCoeff;
    probParams.get("L_1", lightCoeff);
    cudaMalloc(&solverInput.d_litcoeff, sizeof(float)*9);
    cudaMemcpy(solverInput.d_litcoeff, lightCoeff.ptr, sizeof(float) * 9, cudaMemcpyHostToDevice);
    
    //solverInput.deltaTransform = rawSolverInput.parameters.deltaTransform; // transformation to last frame, unused
    solverInput.calibparams.ux = getTypedParameter<float>("u_x", probParams);
    solverInput.calibparams.uy = getTypedParameter<float>("u_y", probParams);
    solverInput.calibparams.fx = getTypedParameter<float>("f_x", probParams);
    solverInput.calibparams.fy = getTypedParameter<float>("f_y", probParams);

    SolverParameters parameters;
    parameters.weightFitting            = getTypedParameter<float>("w_p", probParams);
    parameters.weightShadingStart       = getTypedParameter<float>("w_g", probParams);
    parameters.weightShadingIncrement   = 0.0f; // unused
    parameters.weightShading            = parameters.weightShadingStart;
    parameters.weightRegularizer        = getTypedParameter<float>("w_s", probParams); //rawSolverInput.parameters.weightRegularizer;
    parameters.weightBoundary           = 0.0f; //unused rawSolverInput.parameters.weightBoundary;
    parameters.weightPrior              = 0.0f;//unused rawSolverInput.parameters.weightPrior;
    parameters.nNonLinearIterations     = getTypedParameter<unsigned int>("nonLinearIterations", solverParams);
    parameters.nLinIterations           = getTypedParameter<unsigned int>("linearIterations", solverParams);
    parameters.nPatchIterations         = 1; //unused rawSolverInput.parameters.nPatchIterations;
	
    ConvergenceAnalysis<float>* ca = NULL;
    return solveSFSStub(solverInput, m_solverState, parameters, ca);
}
