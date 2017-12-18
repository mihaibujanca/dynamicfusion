#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include "../../shared/cudaUtil.h"

#include "CUDAImageSolver.h"

#include "CeresImageSolver.h"
#include "SFSSolverInput.h"
#include "../../shared/SolverIteration.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"



class CombinedSolver : public CombinedSolverBase {
private:
    std::shared_ptr<SimpleBuffer>   m_initialUnknown;
    std::shared_ptr<SimpleBuffer>   m_result;
    std::vector<unsigned int> m_dims;
public:
    CombinedSolver(const SFSSolverInput& inputGPU, CombinedSolverParameters params)
	{
        m_combinedSolverParameters = params;
        m_initialUnknown = std::make_shared<SimpleBuffer>(*inputGPU.initialUnknown, true);
        m_result = std::make_shared<SimpleBuffer>(*inputGPU.initialUnknown, true);
        inputGPU.setParameters(m_problemParams, m_result);

        m_dims = { (unsigned int)m_result->width(), (unsigned int)m_result->height() };

        addSolver(std::make_shared<CUDAImageSolver>(m_dims), "CUDA", m_combinedSolverParameters.useCUDA);
        addOptSolvers(m_dims, "shape_from_shading.t", m_combinedSolverParameters.optDoublePrecision);
        addSolver(std::make_shared<CeresImageSolver>(m_dims), "Ceres", m_combinedSolverParameters.useCeres);
	}

    virtual void combinedSolveInit() override {
        m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
        m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
    }

    virtual void preSingleSolve() override {
        resetGPUMemory();
    }
    virtual void postSingleSolve() override {}

    virtual void preNonlinearSolve(int) override {}
    virtual void postNonlinearSolve(int) override {}

    virtual void combinedSolveFinalize() override {
        ceresIterationComparison("Shape From Shading", m_combinedSolverParameters.optDoublePrecision);
    }

    std::shared_ptr<SimpleBuffer> result() {
        return m_result;
    }

	void resetGPUMemory() {
        cudaSafeCall(cudaMemcpy(m_result->data(), m_initialUnknown->data(), m_dims[0]*m_dims[1]*sizeof(float), cudaMemcpyDeviceToDevice));
	}

};
