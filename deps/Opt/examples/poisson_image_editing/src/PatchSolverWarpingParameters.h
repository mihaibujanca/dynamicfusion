#pragma once

#ifndef _PATCH_SOLVER_PARAMETERS_
#define _PATCH_SOLVER_PARAMETERS_

struct PatchSolverParameters
{
	float weightFitting;					// Fitting weights
	float weightRegularizer;				// Regularization weight

	unsigned int nNonLinearIterations;		// Steps of the non-linear solver	
	unsigned int nLinearIterations;			// Steps of the linear solver
	unsigned int nPatchIterations;			// Steps on linear step on block level
};

#endif
