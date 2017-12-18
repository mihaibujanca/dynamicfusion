#include <iostream>

#include "WarpingSolverParameters.h"
#include "PatchSolverWarpingState.h"
#include "PatchSolverWarpingUtil.h"
#include "PatchSolverWarpingEquations.h"

#include "CUDATimer.h"

// For the naming scheme of the variables see:
// http://en.wikipedia.org/wiki/Conjugate_gradient_method
// This code is an implementation of their PCG pseudo code

/////////////////////////////////////////////////////////////////////////
// PCG Patch Iteration
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
// Eval Residual
/////////////////////////////////////////////////////////////////////////

__global__ void ResetResidualDevice(PatchSolverInput input, PatchSolverState state, PatchSolverParameters parameters)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x == 0) state.d_sumResidual[0] = 0.0f;
}

__global__ void EvalResidualDevice(PatchSolverInput input, PatchSolverState state, PatchSolverParameters parameters)
{
	const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N)
	{
		float4 residual = evalFDevice(x, input, state, parameters);
		float r = warpReduce(residual.x + residual.y + residual.z + residual.w);
		unsigned int laneid;
		//This command gets the lane ID within the current warp
		asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
		if (laneid == 0) {
			atomicAdd(&state.d_sumResidual[0], r);
		}
	}
}

float EvalResidual(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters, CUDATimer& timer)
{
	float residual = 0.0f;

	const unsigned int N = input.N; // Number of block variables
	ResetResidualDevice << < 1, 1, 1 >> >(input, state, parameters);
	cudaSafeCall(cudaDeviceSynchronize());
	timer.startEvent("EvalResidual");
	EvalResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
	timer.endEvent();
	cudaSafeCall(cudaDeviceSynchronize());

	residual = state.getSumResidual();

#ifdef _DEBUG
	cudaSafeCall(cudaDeviceSynchronize());
#endif

	return residual;
}

__global__ void PCGStepPatch_Kernel(PatchSolverInput input, PatchSolverState state, PatchSolverParameters parameters, int ox, int oy)
{
	const unsigned int W = input.width;
	const unsigned int H = input.height;

	const int tId_j = threadIdx.x; // local col idx
	const int tId_i = threadIdx.y; // local row idx

	const int gId_j = blockIdx.x * blockDim.x + threadIdx.x - ox; // global col idx
	const int gId_i = blockIdx.y * blockDim.y + threadIdx.y - oy; // global row idx
	
	//////////////////////////////////////////////////////////////////////////////////////////
	// CACHE data to shared memory
	//////////////////////////////////////////////////////////////////////////////////////////
	
	__shared__ float4 X[SHARED_MEM_SIZE_PATCH]; loadPatchToCache(X, state.d_x,	    tId_i, tId_j, gId_i, gId_j, W, H);
	__shared__ float4 T[SHARED_MEM_SIZE_PATCH]; loadPatchToCache(T, state.d_target, tId_i, tId_j, gId_i, gId_j, W, H);
	__shared__ float  M[SHARED_MEM_SIZE_PATCH]; loadPatchToCache(M, state.d_mask,	tId_i, tId_j, gId_i, gId_j, W, H);

	__shared__ float4 P [SHARED_MEM_SIZE_PATCH]; setPatchToZero(P,  tId_i, tId_j);

	__shared__ float patchBucket[SHARED_MEM_SIZE_VARIABLES];

	__syncthreads();

	//////////////////////////////////////////////////////////////////////////////////////////
	// CACHE data to registers
	//////////////////////////////////////////////////////////////////////////////////////////

	register float4 X_CC  = readValueFromCache2D(X, tId_i, tId_j);
	register float  M_CC  = readValueFromCache2D(M, tId_i, tId_j);
	register bool   isValidPixel = isValid(X_CC) && M_CC == 0;

	register float4 Delta  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	register float4 R;
	register float4 Z;
	register float4 Pre;
	register float  RDotZOld;
	register float4 AP;
	
	__syncthreads();
	
	//////////////////////////////////////////////////////////////////////////////////////////
	// Initialize linear patch systems
	//////////////////////////////////////////////////////////////////////////////////////////

	float d = 0.0f;
	if (isValidPixel)
	{
		R = evalMinusJTFDevice(tId_i, tId_j, gId_i, gId_j, W, H, M, T, X, parameters, Pre); // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 
		float4 preRes  = Pre *R;															// apply preconditioner M^-1
		P [getLinearThreadIdCache(tId_i, tId_j)] = preRes;									// save for later
	
		d = dot(R, preRes);
	}
	
	patchBucket[getLinearThreadId(tId_i, tId_j)] = d;										 // x-th term of nomimator for computing alpha and denominator for computing beta
	
	__syncthreads();
	blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES);
	__syncthreads();
	
	if (isValidPixel) RDotZOld = patchBucket[0];							   // read result for later on
	
	__syncthreads();
	
	////////////////////////////////////////////////////////////////////////////////////////////
	//// Do patch PCG iterations
	////////////////////////////////////////////////////////////////////////////////////////////
	
	for(unsigned int patchIter = 0; patchIter < parameters.nPatchIterations; patchIter++)
	{
		const float4 currentP  = P [getLinearThreadIdCache(tId_i, tId_j)];
	
		float d = 0.0f;
		if (isValidPixel)
		{
			AP = applyJTJDevice(tId_i, tId_j, gId_i, gId_j, W, H, M, T, P, X, parameters);	// A x p_k  => J^T x J x p_k 
			d = dot(currentP, AP);															// x-th term of denominator of alpha
		}
	
		patchBucket[getLinearThreadId(tId_i, tId_j)] = d;
	
		__syncthreads();
		blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES);
		__syncthreads();
		
		const float dotProduct = patchBucket[0];
	
		float b = 0.0f;
		if (isValidPixel)
		{
			float alpha = 0.0f;
			if(dotProduct > FLOAT_EPSILON) alpha = RDotZOld/dotProduct;	    // update step size alpha
			Delta  = Delta  + alpha*currentP;								// do a decent step		
			R  = R  - alpha*AP;												// update residuum	
			Z  = Pre *R;													// apply preconditioner M^-1
			b = dot(Z,R);													// compute x-th term of the nominator of beta
		}
	
		__syncthreads();													// Only write if every thread in the block has has read bucket[0]
	
		patchBucket[getLinearThreadId(tId_i, tId_j)] = b;
	
		__syncthreads();
		blockReduce(patchBucket, getLinearThreadId(tId_i, tId_j), SHARED_MEM_SIZE_VARIABLES);	// sum over x-th terms to compute nominator of beta inside this block
		__syncthreads();
	
		if (isValidPixel)
		{
			const float rDotzNew = patchBucket[0];												// get new nominator
			
			float beta = 0.0f;														 
			if(RDotZOld > FLOAT_EPSILON) beta = rDotzNew/RDotZOld;								// update step size beta
			RDotZOld = rDotzNew;																// save new rDotz for next iteration
			P [getLinearThreadIdCache(tId_i, tId_j)] = Z + beta*currentP;						// update decent direction
		}
	
		__syncthreads();
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////
	//// Save to global memory
	////////////////////////////////////////////////////////////////////////////////////////////
	
	if (isValidPixel)
	{
		state.d_x[get1DIdx(gId_i, gId_j, W, H)] = X_CC  + Delta;
	}
}

void PCGIterationPatch(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters, int ox, int oy, CUDATimer& timer)
{
	dim3 blockSize(PATCH_SIZE, PATCH_SIZE);
	dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x + 1, (input.height + blockSize.y - 1) / blockSize.y + 1); // one more for block shift!

	cudaSafeCall(cudaDeviceSynchronize());
	timer.startEvent("PCGIterationPatch");
	PCGStepPatch_Kernel<<<gridSize, blockSize>>>(input, state, parameters, ox, oy);
	timer.endEvent();

	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif
}

////////////////////////////////////////////////////////////////////
// Main GN Solver Loop
////////////////////////////////////////////////////////////////////

int offsetX[8] = {(int)(0.0f*PATCH_SIZE), (int)((1.0f/2.0f)*PATCH_SIZE), (int)((1.0f/4.0f)*PATCH_SIZE), (int)((3.0f/4.0f)*PATCH_SIZE), (int)((1.0f/8.0f)*PATCH_SIZE), (int)((5.0f/8.0f)*PATCH_SIZE), (int)((3.0f/8.0f)*PATCH_SIZE), (int)((7.0f/8.0f)*PATCH_SIZE)}; // Halton sequence base 2
int offsetY[8] = {(int)(0.0f*PATCH_SIZE), (int)((1.0f/3.0f)*PATCH_SIZE), (int)((2.0f/3.0f)*PATCH_SIZE), (int)((1.0f/9.0f)*PATCH_SIZE), (int)((4.0f/9.0f)*PATCH_SIZE), (int)((7.0f/9.0f)*PATCH_SIZE), (int)((2.0f/9.0f)*PATCH_SIZE), (int)((5.0f/9.0f)*PATCH_SIZE)}; // Halton sequence base 3

extern "C" double patchSolveStereoStub(PatchSolverInput& input, PatchSolverState& state, PatchSolverParameters& parameters)
{
	CUDATimer timer;

	int o = 0;
	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{	
		float residual = EvalResidual(input, state, parameters, timer);
		printf("%i: cost: %f\n", nIter, residual);

		for (unsigned int lIter = 0; lIter < parameters.nLinearIterations; lIter++) {
			PCGIterationPatch(input, state, parameters, offsetX[o], offsetY[o], timer);
			o = (o + 1) % 8;
		}
		timer.nextIteration();
	}
	timer.evaluate();

	float residual = EvalResidual(input, state, parameters, timer);
	printf("final cost: %f\n", residual);
    return (double)residual;
}
