#pragma once

#include <cuda_runtime.h>

#include "../../shared/cudaUtil.h"
#include "SFSSolverParameters.h"
#include "SFSSolverState.h"
#include "../../shared/SolverBase.h"

#include <memory>
#include "SimpleBuffer.h"
#include "SFSSolverInput.h"
class CUDAImageSolver : public SolverBase
{
	public:
        CUDAImageSolver(const std::vector<unsigned int>& dims);
		~CUDAImageSolver();

        virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters) override;
		
	private:

		SolverState	m_solverState;
        std::vector<unsigned int> m_dims;
};
