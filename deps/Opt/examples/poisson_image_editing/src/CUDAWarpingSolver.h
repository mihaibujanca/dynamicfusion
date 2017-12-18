#pragma once

#include <cuda_runtime.h>

#include "../../shared/cudaUtil.h"
#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"
#include "../../shared/SolverBase.h"

class CUDAWarpingSolver : public SolverBase
{
	public:
        CUDAWarpingSolver(const std::vector<unsigned int>& dims);
		~CUDAWarpingSolver();

        virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters) override;
		
	private:

		SolverState	m_solverState;
        std::vector<unsigned int> m_dims;
};
