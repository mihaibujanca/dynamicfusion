#include "CUDAWarpingSolver.h"
#include "../../shared/OptUtils.h"
#include "../../shared/OptSolver.h"

extern "C" double ImageWarpingSolveGNStub(SolverInput& input, SolverState& state, SolverParameters& parameters);	// gauss newton



CUDAWarpingSolver::CUDAWarpingSolver(const std::vector<unsigned int>& dims) : m_dims(dims)
{

    const unsigned int numberOfVariables = dims[0] * dims[1] * dims[2];

	// State
	cudaSafeCall(cudaMalloc(&m_solverState.d_delta,		sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_deltaA,		sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_r,			sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_rA,			sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_z,			sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_zA,			sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_p,			sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_pA,			sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_Ap_X,			sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_Ap_A,			sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_scanAlpha,	sizeof(float)));
	cudaSafeCall(cudaMalloc(&m_solverState.d_scanBeta,		sizeof(float)));
	cudaSafeCall(cudaMalloc(&m_solverState.d_rDotzOld,		sizeof(float)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_precondioner, sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_precondionerA, sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_sumResidual,	sizeof(float)));
}

CUDAWarpingSolver::~CUDAWarpingSolver()
{
	// State
	cudaSafeCall(cudaFree(m_solverState.d_delta));
	cudaSafeCall(cudaFree(m_solverState.d_deltaA));
	cudaSafeCall(cudaFree(m_solverState.d_r));
	cudaSafeCall(cudaFree(m_solverState.d_rA));
	cudaSafeCall(cudaFree(m_solverState.d_z));
	cudaSafeCall(cudaFree(m_solverState.d_zA));
	cudaSafeCall(cudaFree(m_solverState.d_p));
	cudaSafeCall(cudaFree(m_solverState.d_pA));
	cudaSafeCall(cudaFree(m_solverState.d_Ap_X));
	cudaSafeCall(cudaFree(m_solverState.d_Ap_A));
	cudaSafeCall(cudaFree(m_solverState.d_scanAlpha));
	cudaSafeCall(cudaFree(m_solverState.d_scanBeta));
	cudaSafeCall(cudaFree(m_solverState.d_rDotzOld));
	cudaSafeCall(cudaFree(m_solverState.d_precondioner));
	cudaSafeCall(cudaFree(m_solverState.d_precondionerA));
	cudaSafeCall(cudaFree(m_solverState.d_sumResidual));
}

float sq(float x) { return x*x; }

double CUDAWarpingSolver::solve(const NamedParameters& solverParams, const NamedParameters& probParams, bool profileSolve, std::vector<SolverIteration>& iters)
{

    m_solverState.d_urshape = getTypedParameterImage<float3>("UrShape", probParams);
    m_solverState.d_a = getTypedParameterImage<float3>("Angle", probParams);
    m_solverState.d_target = getTypedParameterImage<float3>("Constraints", probParams);
    m_solverState.d_x = getTypedParameterImage<float3>("Offset", probParams);


	SolverParameters parameters;
    parameters.weightFitting = sq(getTypedParameter<float>("w_fitSqrt", probParams));
    parameters.weightRegularizer = sq(getTypedParameter<float>("w_regSqrt", probParams));
    parameters.nNonLinearIterations = getTypedParameter<unsigned int>("nIterations", solverParams);
    parameters.nLinIterations = getTypedParameter<unsigned int>("lIterations", solverParams);
	
	SolverInput solverInput;
    solverInput.N = m_dims[0] * m_dims[1] * m_dims[2];
    solverInput.dims = make_int3(m_dims[0] - 1, m_dims[1] - 1, m_dims[2] - 1);

	return ImageWarpingSolveGNStub(solverInput, m_solverState, parameters);
}
