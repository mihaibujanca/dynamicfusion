#pragma once

#ifndef _SOLVER_PARAMETERS_
#define _SOLVER_PARAMETERS_

struct SolverParameters
{
    float weightFitting;					// Is initialized by the solver!

    float weightRegularizer;				// Regularization weight
    float weightPrior;						// Prior weight

    float weightShading;					// Shading weight
    float weightShadingStart;				// Starting value for incremental relaxation
    float weightShadingIncrement;			// Update factor

    float weightBoundary;					// Boundary weight

    unsigned int nNonLinearIterations;		// Steps of the non-linear solver	
    unsigned int nLinIterations;			// Steps of the linear solver
    // Unused for non-patch solver
    unsigned int nPatchIterations;			// Steps on linear step on block level
};

#endif
