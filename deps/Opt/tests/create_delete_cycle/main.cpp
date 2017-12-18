extern "C" {
#include "Opt.h"
}
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../shared/stb_image_write.h"

void solveLaplacian(int width, int height, float* unknown, float* target) {
    Opt_InitializationParameters param = {};
    param.doublePrecision = 0;
    param.verbosityLevel = 1;
    param.collectPerKernelTimingInfo = 1;
    Opt_State* state = Opt_NewState(param);


    // load the Opt DSL file containing the cost description
    Opt_Problem* problem = Opt_ProblemDefine(state, "laplacian.t", "gaussNewtonGPU");
    // describe the dimensions of the instance of the problem
    unsigned int dims[] = { width, height };
    for (int i = 0; i < 1000; ++i) {
        std::cout << "Iteration: " << i << std::endl;
        Opt_Plan* plan = Opt_ProblemPlan(state, problem, dims);
        Opt_PlanFree(state, plan);
    }
    Opt_Plan* plan = Opt_ProblemPlan(state, problem, dims);
    // run the solver
    void* problem_data[] = { unknown, target };
    Opt_ProblemSolve(state, plan, problem_data);
    Opt_ProblemDelete(state, problem);
}

void saveMonochromeImage(char const * filename, const int width, const int height, float* d_data) {
    float *data = new float[width*height];
    cudaMemcpy(data, d_data, width*height*sizeof(float), cudaMemcpyDeviceToHost);

    unsigned char* convertedData = new unsigned char[width*height];
    for (int i = 0; i < width*height; ++i) {
        convertedData[i] = (unsigned char)(data[i] * 255);
    }
    stbi_write_png(filename, width, height, 1, convertedData, width);

    delete[] data;
    delete[] convertedData;
}


int main(){
    const int dim = 512;
    float* scratch = new float[dim*dim];
    for (int i = 0; i < dim*dim; ++i) {
        scratch[i] = (float)((double)rand() / (double)RAND_MAX);
    }
    float *target, *unknown;
    size_t fSize = dim*dim*sizeof(float);
    cudaMalloc(&target, fSize);
    cudaMalloc(&unknown, fSize);
    cudaMemcpy(target, scratch, fSize, cudaMemcpyHostToDevice);
    cudaMemcpy(unknown, target, fSize, cudaMemcpyDeviceToDevice);
    delete[] scratch;

    solveLaplacian(dim, dim, unknown, target);
    saveMonochromeImage("target.png", dim, dim, target);
    saveMonochromeImage("result.png", dim, dim, unknown);

    cudaFree(target);
    cudaFree(unknown);
    return 0;
}