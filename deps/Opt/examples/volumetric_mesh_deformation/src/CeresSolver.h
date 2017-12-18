#pragma once

#include <cuda_runtime.h>

#include "mLibInclude.h"
#include "../../shared/cudaUtil.h"
#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"
#include "../../shared/SolverIteration.h"
#include "../../shared/CeresSolverBase.h"
#include "../../shared/Config.h"
#if USE_CERES
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
#endif

class CeresSolver : public CeresSolverBase
{
	public:
        CeresSolver(const std::vector<unsigned int>& dims) : CeresSolverBase(dims) {}
        virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) override;
};
#if !USE_CERES
inline double CeresSolver::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) {
    return nan("");
}
#endif