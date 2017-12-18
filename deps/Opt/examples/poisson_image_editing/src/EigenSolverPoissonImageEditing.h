#pragma once
#include "../../shared/SolverBase.h"
class EigenSolverPoissonImageEditing : public SolverBase {

public:
    EigenSolverPoissonImageEditing(const std::vector<unsigned int>& dims) : m_dims(dims) {}

    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters) override;

private:
    std::vector<unsigned int> m_dims;
};
