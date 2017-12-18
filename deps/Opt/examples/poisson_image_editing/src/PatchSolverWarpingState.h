#pragma once

#ifndef _PATCH_SOLVER_STATE_
#define _PATCH_SOLVER_STATE_

#include <cuda_runtime.h> 

struct PatchSolverInput
{
	// Size of optimization domain
	unsigned int N;					// Number of variables

	unsigned int width;				// Image width
	unsigned int height;			// Image height
};

struct PatchSolverState
{
	// State of the GN Solver
	float4*  d_x;
	float4*  d_target;
	float*	 d_mask;

	float*	d_sumResidual;

	__host__ float getSumResidual() const {
		float residual;
		cudaMemcpy(&residual, d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost);
		return residual;
	}
};

#endif
