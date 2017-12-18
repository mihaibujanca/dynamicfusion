#pragma once

#include <cuda_runtime.h> 
#include "CameraParams.h"
#include "../../shared/cuda_SimpleMatrixUtil.h"

struct SolverInput
{
    // Size of optimization domain
    unsigned int N;					// Number of variables

    unsigned int width;				// Image width
    unsigned int height;			// Image height

    // Target
    float* d_targetIntensity;		// Constant target values
    float* d_targetDepth;			// Constant target values

    float* d_depthMapRefinedLastFrameFloat; // refined result of last frame

    // mask edge map
    unsigned char* d_maskEdgeMapR;
    unsigned char* d_maskEdgeMapC;

    // Lighting
    float* d_litcoeff;

    //camera intrinsic parameter
    CameraParams calibparams;

    float4x4 deltaTransform;

};



struct SolverState
{
    // State of the GN Solver
    float*	d_delta;
    float*  d_x;

    float*	d_r;
    float*	d_z;
    float*	d_p;

    float*	d_Ap_X;

    float*	d_scanAlpha;
    float*	d_scanBeta;
    float*	d_rDotzOld;

    float*	d_preconditioner;

    float*	d_sumResidual;

    // Precompute buffers
    float* B_I;
    float* B_I_dx0;
    float* B_I_dx1;
    float* B_I_dx2;
    bool* pguard;

};
