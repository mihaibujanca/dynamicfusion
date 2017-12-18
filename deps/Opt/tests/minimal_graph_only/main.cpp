extern "C" {
#include "Opt.h"
}
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define OPT_DOUBLE_PRECISION 1

#if OPT_DOUBLE_PRECISION
#define OPT_FLOAT double
#define OPT_FLOAT2 double2
#else
#define OPT_FLOAT float
#define OPT_FLOAT2 float2
#endif

void solve(int dataCount, int* startNodes, int* endNodes, OPT_FLOAT* params, OPT_FLOAT* data, std::string name) {
    Opt_InitializationParameters param = {};
    param.doublePrecision = OPT_DOUBLE_PRECISION;
    param.verbosityLevel = 2;
    param.collectPerKernelTimingInfo = 1;
    //param.threadsPerBlock = 512;
    Opt_State* state = Opt_NewState(param);
    // load the Opt DSL file containing the cost description
    Opt_Problem* problem = Opt_ProblemDefine(state, name.c_str(), "gaussNewtonGPU");
    // describe the dimensions of the instance of the problem
    unsigned int dims[] = { dataCount, 1 };
    Opt_Plan* plan = Opt_ProblemPlan(state, problem, dims);
    // run the solver
    void* problem_data[] = { params, data, &dataCount, endNodes, startNodes };
    Opt_ProblemSolve(state, plan, problem_data);
    Opt_PlanFree(state, plan);
    Opt_ProblemDelete(state, problem);
}


int main(){

    const int dim = 512;
    OPT_FLOAT2 generatorParams = { 100.0, 102.0 };
    std::vector<OPT_FLOAT2> dataPoints(dim);
    OPT_FLOAT a = generatorParams.x;
    OPT_FLOAT b = generatorParams.y;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-50.0, 50.0);
    for (int i = 0; i < dataPoints.size(); ++i) {
        OPT_FLOAT x = float(i)*2.0*3.141592653589 / dim;
        OPT_FLOAT y = (a*cos(b*x) + b*sin(a*x));
        //y = a*x + b;
        // Add in noise
        //y += dis(gen);
        dataPoints[i].x = x;
        dataPoints[i].y = y;

    }
    
    OPT_FLOAT2 unknownInit = { 99.7f, 101.6f };

    OPT_FLOAT *d_data, *d_unknown;
    cudaMalloc(&d_data, dim*sizeof(OPT_FLOAT2));
    cudaMalloc(&d_unknown, sizeof(OPT_FLOAT2));
    cudaMemcpy(d_data, dataPoints.data(), dim*sizeof(OPT_FLOAT2), cudaMemcpyHostToDevice);
    

    int *d_startNodes, *d_endNodes;
    cudaMalloc(&d_startNodes, dim*sizeof(int));
    cudaMalloc(&d_endNodes, dim*sizeof(int));


    std::vector<int> endNodes;
    for (int i = 0; i < dim; ++i) { endNodes.push_back(i); }
    cudaMemset(d_startNodes, 0, dim*sizeof(int));
    cudaMemcpy(d_endNodes, endNodes.data(), dim*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_unknown, &unknownInit, sizeof(OPT_FLOAT2), cudaMemcpyHostToDevice);
    solve(dim, d_startNodes, d_endNodes, d_unknown, d_data, "curveFitting.t");


    OPT_FLOAT2 unknownResult = {};
    cudaMemcpy(&unknownResult, d_unknown, sizeof(OPT_FLOAT2), cudaMemcpyDeviceToHost);

    

    std::cout << "Init " << unknownInit.x << ", " << unknownInit.y << std::endl;
    std::cout << "Result " << unknownResult.x << ", " << unknownResult.y << std::endl;
    std::cout << "Goal " << generatorParams.x << ", " << generatorParams.y << std::endl;

    cudaFree(d_data);
    cudaFree(d_unknown);
    cudaFree(d_startNodes);
    cudaFree(d_endNodes);
    return 0;
}