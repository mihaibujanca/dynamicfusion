#pragma once

#include "../../shared/cudaUtil.h"
#include "../../shared/cuda_SimpleMatrixUtil.h"

#define FLOAT_EPSILON 0.0001f

#ifndef BYTE
#define BYTE unsigned char
#endif

#define MINF __int_as_float(0xff800000)

///////////////////////////////////////////////////////////////////////////////
// Helper ARAP
///////////////////////////////////////////////////////////////////////////////

// Rotation Matrix
inline __device__ float2x2 evalR(float CosAlpha, float SinAlpha)
{
	float2x2 R;
	R.m11 = CosAlpha; R.m12 = -SinAlpha;
	R.m21 = SinAlpha; R.m22 =  CosAlpha;
	return R;
}

inline __device__ float2x2 evalR(float& angle)
{
	return evalR(cos(angle), sin(angle));
}

// Rotation Matrix dAlpha
inline __device__ float2x2 eval_dR(float CosAlpha, float SinAlpha)
{
	float2x2 R;
	R.m11 = -SinAlpha; R.m12 = -CosAlpha;
	R.m21 = CosAlpha;  R.m22 = -SinAlpha;
	return R;
}

inline __device__ float2x2 evalR_dR(float& angle)
{
	return eval_dR(cos(angle), sin(angle));
}

///////////////////////////////////////////////////////////////////////////////
// Helper
///////////////////////////////////////////////////////////////////////////////

__inline__ __device__ void get2DIdx(int idx, unsigned int width, unsigned int height, int& i, int& j)
{
	i = idx / width;
	j = idx % width;
}

__inline__ __device__ unsigned int get1DIdx(int i, int j, unsigned int width, unsigned int height)
{
	return i*width+j;
}

__inline__ __device__ bool isInsideImage(int i, int j, unsigned int width, unsigned int height)
{
	return (i >= 0 && i < height && j >= 0 && j < width);
}

__inline__ __device__ bool inLaplacianBounds(int i, int j, unsigned int width, unsigned int height)
{
    return (i > 0 && i < (height-1) && j > 0 && j < (width - 1));
}

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

