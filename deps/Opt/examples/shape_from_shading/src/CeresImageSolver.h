#pragma once

#include <cassert>

#include "SimpleBuffer.h"
#include "SFSSolverInput.h"
#include "../../shared/SolverIteration.h"
#include "../../shared/CeresSolverBase.h"
#include "../../shared/Config.h"
class CeresImageSolver : public CeresSolverBase {

public:
    CeresImageSolver(const std::vector<unsigned int>& dims) : CeresSolverBase(dims) {}

    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters) override;

    inline int getPixel(int x, int y) const
    {
        return y * (int)m_dims[0] + x;
    }
    std::vector<float> Xfloat;
    std::vector<float> D_i;
    std::vector<float> Im;
    std::vector<BYTE>  edgeMaskR;
    std::vector<BYTE>  edgeMaskC;
    float f_x;
    float f_y;
    float u_x;
    float u_y;
    float L[9];
};

#if !USE_CERES
double CeresImageSolver::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters) { return nan(""); }
#endif