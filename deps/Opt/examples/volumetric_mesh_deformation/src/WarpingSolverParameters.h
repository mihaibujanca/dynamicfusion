#pragma once

#ifndef _SOLVER_PARAMETERS_
#define _SOLVER_PARAMETERS_

struct SolverParameters
{
	float weightFitting;					// Fitting weights
	float weightRegularizer;				// Regularization weight

	unsigned int nNonLinearIterations;		// Steps of the non-linear solver	
	unsigned int nLinIterations;			// Steps of the linear solver
};

#endif
