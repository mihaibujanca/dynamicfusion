#include "CUDAWarpingSolver.h"
#include "../../shared/OptUtils.h"
extern "C" double ImageWarpingSolveGNStub(SolverInput& input, SolverState& state, SolverParameters& parameters);	// gauss newton

CUDAWarpingSolver::CUDAWarpingSolver(const std::vector<unsigned int>& dims) : m_dims(dims)
{
	const unsigned int THREADS_PER_BLOCK = 1024; // keep consistent with the GPU
	const unsigned int tmpBufferSize = THREADS_PER_BLOCK*THREADS_PER_BLOCK;

    const unsigned int N = dims[0] * dims[1];
	const unsigned int numberOfVariables = N;

	// State
	cudaSafeCall(cudaMalloc(&m_solverState.d_delta,		sizeof(float4)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_r,			sizeof(float4)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_z,			sizeof(float4)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_p,			sizeof(float4)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_Ap_X,			sizeof(float4)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_scanAlpha,	sizeof(float)*tmpBufferSize));
	cudaSafeCall(cudaMalloc(&m_solverState.d_scanBeta,		sizeof(float)*tmpBufferSize));
	cudaSafeCall(cudaMalloc(&m_solverState.d_rDotzOld,		sizeof(float)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_precondioner, sizeof(float4)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_sumResidual,	sizeof(float)));
}

CUDAWarpingSolver::~CUDAWarpingSolver()
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
	cudaSafeCall(cudaFree(m_solverState.d_precondioner));
	cudaSafeCall(cudaFree(m_solverState.d_sumResidual));
}

double CUDAWarpingSolver::solve(const NamedParameters& solverParams, const NamedParameters& probParams, bool profileSolve, std::vector<SolverIteration>& iters)
{
    m_solverState.d_target  = getTypedParameterImage<float4>("T", probParams);
    m_solverState.d_mask    = getTypedParameterImage<float>("M", probParams);
    m_solverState.d_x       = getTypedParameterImage<float4>("X", probParams);

	SolverParameters parameters;
    parameters.nNonLinearIterations = getTypedParameter<unsigned int>("nonLinearIterations", solverParams);
    parameters.nLinIterations = getTypedParameter<unsigned int>("linearIterations", solverParams);
	
	SolverInput solverInput;
    solverInput.N = m_dims[0] * m_dims[1];
    solverInput.width = m_dims[0];
    solverInput.height = m_dims[1];

	return ImageWarpingSolveGNStub(solverInput, m_solverState, parameters);
}
