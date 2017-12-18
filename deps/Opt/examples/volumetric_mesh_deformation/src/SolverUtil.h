#pragma once

#include "../../shared/cudaUtil.h"
#include "../../shared/cuda_SimpleMatrixUtil.h"

#define FLOAT_EPSILON 0.0f

#ifndef BYTE
#define BYTE unsigned char
#endif

#define MINF __int_as_float(0xff800000)

///////////////////////////////////////////////////////////////////////////////
// Helper
///////////////////////////////////////////////////////////////////////////////

inline __device__ void warpReduce(volatile float* sdata, int threadIdx, unsigned int threadsPerBlock) // See Optimizing Parallel Reduction in CUDA by Mark Harris
{
	if(threadIdx < 32)
	{
		if(threadIdx + 32 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx + 32];
		if(threadIdx + 16 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx + 16];
		if(threadIdx +  8 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx +  8];
		if(threadIdx +  4 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx +  4];
		if(threadIdx +  2 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx +  2];
		if(threadIdx +  1 < threadsPerBlock) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx +  1];
	}
}

inline __device__ void blockReduce(volatile float* sdata, int threadIdx, unsigned int threadsPerBlock)
{
	#pragma unroll
	for(unsigned int stride = threadsPerBlock/2 ; stride > 32; stride/=2)
	{
		if(threadIdx < stride) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx+stride];

		__syncthreads();
	}

	warpReduce(sdata, threadIdx, threadsPerBlock);
}
