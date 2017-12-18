#pragma once

#include "SolverUtil.h"

#include "../../shared/cudaUtil.h"

#define THREADS_PER_BLOCK 1024 // keep consistent with the CPU

#define DR_THREAD_SIZE1_X 32
#define DR_THREAD_SIZE1_Y 8

__inline__ __device__ float warpReduce(float val) {
    int offset = 32 >> 1;
    while (offset > 0) {
        val = val + __shfl_down(val, offset, 32);
        offset = offset >> 1;
    }
    return val;
}

extern __shared__ float bucket[];

__inline__ __device__ bool getGlobalNeighbourIdxFromLocalNeighourIdx(int centerIdx, int localNeighbourIdx, const SolverInput& input, int& outGlobalNeighbourIdx)
{
	int i;	int j; get2DIdx(centerIdx, input.width, input.height, i, j);

	const int d = ((int)(2*(localNeighbourIdx % 2)))-1;
	const int m = localNeighbourIdx/2;

	const int iNeigh = ((int)i) + m*d;
	const int jNeigh = ((int)j) + (1-m)*d;

	if(!isInsideImage(iNeigh, jNeigh, input.width, input.height)) return false;

	outGlobalNeighbourIdx = get1DIdx(iNeigh, jNeigh, input.width, input.height);
	return true;
}

inline __device__ void scanPart1(unsigned int threadIdx, unsigned int blockIdx, unsigned int threadsPerBlock, float* d_output)
{
	__syncthreads();
	blockReduce(bucket, threadIdx, threadsPerBlock);
	if(threadIdx == 0) d_output[blockIdx] = bucket[0];
}

inline __device__ void scanPart2(unsigned int threadIdx, unsigned int threadsPerBlock, unsigned int blocksPerGrid, float* d_tmp)
{
	if(threadIdx < blocksPerGrid) bucket[threadIdx] = d_tmp[threadIdx];
	else						  bucket[threadIdx] = 0.0f;
	
	__syncthreads();
	blockReduce(bucket, threadIdx, threadsPerBlock);
	__syncthreads();
}
