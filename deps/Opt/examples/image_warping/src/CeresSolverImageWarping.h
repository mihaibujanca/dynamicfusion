#pragma once
#include "../../shared/CeresSolverBase.h"
#include "../../shared/Config.h"
class CeresSolverWarping : public CeresSolverBase {
public:
    CeresSolverWarping(const std::vector<unsigned int>& dims) : CeresSolverBase(dims) {}
    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) override;
};
#if !USE_CERES
double CeresSolverWarping::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter)
{
    return nan("");
}
#endif