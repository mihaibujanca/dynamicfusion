#include "CUDAPatchSolverWarping.h"
#include "PatchSolverWarpingParameters.h"
#include "../../shared/OptUtils.h"
extern "C" double patchSolveStereoStub(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters);

CUDAPatchSolverWarping::CUDAPatchSolverWarping(const std::vector<unsigned int>& dims) : m_dims(dims)
{
	cudaSafeCall(cudaMalloc(&m_solverState.d_sumResidual, sizeof(float)));
}

CUDAPatchSolverWarping::~CUDAPatchSolverWarping()
{
	cudaSafeCall(cudaFree(m_solverState.d_sumResidual));
}

double CUDAPatchSolverWarping::solve(const NamedParameters& solverParams, const NamedParameters& probParams, bool profileSolve, std::vector<SolverIteration>& iters)
{
    // TOOD: move this to a more visible place
    unsigned int patchIter = 16;

    m_solverState.d_target = getTypedParameterImage<float4>("T", probParams);
    m_solverState.d_mask = getTypedParameterImage<float>("M", probParams);
    m_solverState.d_x = getTypedParameterImage<float4>("X", probParams);

    PatchSolverParameters parameters;
    parameters.nNonLinearIterations = getTypedParameter<unsigned int>("nonLinearIterations", solverParams);
    parameters.nLinearIterations    = getTypedParameter<unsigned int>("linearIterations", solverParams);
    parameters.nPatchIterations     = patchIter;

	PatchSolverInput solverInput;
    solverInput.N = m_dims[0] * m_dims[1];
    solverInput.width = m_dims[0];
    solverInput.height = m_dims[1];
	
	return patchSolveStereoStub(solverInput, m_solverState, parameters);
}
