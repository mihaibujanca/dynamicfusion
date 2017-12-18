#pragma once

#ifndef _SOLVER_STATE_
#define _SOLVER_STATE_

#include <cuda_runtime.h> 

#ifndef MINF
#ifdef __CUDACC__
#define MINF __int_as_float(0xff800000)
#else
#define  MINF (std::numeric_limits<float>::infinity())
#endif
#endif 

//#define Stereo_ENABLED
#define LE_THREAD_SIZE 16

struct SolverInput
{
	// Size of optimization domain
	unsigned int N;					// Number of variables

	unsigned int width;				// Image width
	unsigned int height;			// Image height
	
	// Target
	float2* d_constraints;			// Constant target values
};



struct SolverState
{
	// State of the GN Solver
	float2*	 d_delta;					// Current linear update to be computed
	float*	 d_deltaA;					// Current linear update to be computed
	float2*  d_x;						// Current state pos
	float*   d_A;						// Current state pos
	float2*  d_urshape;					// Current state urshape pos
	float*	 d_mask;
	
	float2*	d_r;						// Residuum
	float*	d_rA;						// Residuum
	float2*	d_z;						// Predconditioned residuum
	float*	d_zA;						// Predconditioned residuum
	float2*	d_p;						// Decent direction
	float*	d_pA;						// Decent direction
	
	float2*	d_Ap_X;						// Cache values for next kernel call after A = J^T x J x p
	float*	d_Ap_XA;						// Cache values for next kernel call after A = J^T x J x p

	float*	d_scanAlpha;				// Tmp memory for alpha scan
	float*	d_scanBeta;					// Tmp memory for beta scan
	float*	d_rDotzOld;					// Old nominator (denominator) of alpha (beta)
	
	float2*	d_precondioner;				// Preconditioner for linear system
	float*	d_precondionerA;				// Preconditioner for linear system

	float*	d_sumResidual;				// sum of the squared residuals

	__host__ float getSumResidual() const {
		float residual;
		cudaMemcpy(&residual, d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost);
		return residual;
	}
};

#endif
