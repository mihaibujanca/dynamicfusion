#pragma once
#include "../../shared/CeresSolverBase.h"
#include "../../shared/Config.h"
class CeresSolverPoissonImageEditing : public CeresSolverBase {
public:
    CeresSolverPoissonImageEditing(const std::vector<unsigned int>& dims) : CeresSolverBase(dims) {}
    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters) override;
};

#if !USE_CERES
double CeresSolverPoissonImageEditing::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters) 
{
    return nan("");
}

#endif