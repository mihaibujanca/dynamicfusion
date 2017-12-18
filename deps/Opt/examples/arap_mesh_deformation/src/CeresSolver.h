#pragma once

#include <vector>
#include "../../shared/SolverIteration.h"
#include "../../shared/CeresSolverBase.h"

class CeresSolver : public CeresSolverBase {
public:
    CeresSolver(const std::vector<unsigned int>& dims, SimpleMesh* _mesh) : CeresSolverBase(dims), m_mesh(_mesh){}

    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) override;

private:
    SimpleMesh *m_mesh;
};

#if !USE_CERES
inline double CeresSolver::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter)
{
    return nan("");
}
#endif