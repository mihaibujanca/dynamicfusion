#pragma once

#include "SolverBase.h"
#include "Config.h"

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
#include <memory>
class CeresSolverBase : public SolverBase {
public:
    CeresSolverBase(const std::vector<unsigned int>& dims) : m_dims(dims) {}

    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) override {
        fprintf(stderr, "No Ceres solve implemented\n");
        return m_finalCost;
    }

protected:
#if USE_CERES
    double launchProfiledSolveAndSummary(const std::unique_ptr<Solver::Options>& options, Problem* problem, bool profileSolve, std::vector<SolverIteration>& iter);
    std::unique_ptr<Solver::Options> initializeOptions(const NamedParameters& solverParameters) const;
#endif
    std::vector<unsigned int> m_dims;
};
