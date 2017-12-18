#pragma once
#include <string>
#include "../../shared/cuda_SimpleMatrixUtil.h" 
#include <stdio.h>
#include <vector>

struct TerraSolverParameters {
    float weightFitting;					// Is initialized by the solver!

    float weightRegularizer;				// Regularization weight
    // Vesitgal
    float weightPrior;						// Prior weight

    float weightShading;					// Shading weight
    // Vesitgal
    float weightShadingStart;				// Starting value for incremental relaxation
    // Vesitgal
    float weightShadingIncrement;			// Update factor
    // Vesitgal
    float weightBoundary;					// Boundary weight

    float fx;
    float fy;
    float ux;
    float uy;

    // Vesitgal
    float4x4 deltaTransform;
    float lightingCoefficients[9];

    unsigned int unusued[3];

    TerraSolverParameters() {}
    void load(const std::string& filename) {
        FILE* fileHandle = fopen(filename.c_str(), "rb"); //b for binary
        fread(this, sizeof(TerraSolverParameters), 1, fileHandle);
        fclose(fileHandle);
    }

    void save(const std::string& filename) {
        FILE* fileHandle = fopen(filename.c_str(), "wb"); //b for binary
        fwrite(this, sizeof(TerraSolverParameters), 1, fileHandle);
        fclose(fileHandle);
    }
};

struct TerraSolverParameterPointers {
    float* floatPointers[36];
    unsigned int* uintPointers[3];
    void* imagePointers[6];
    TerraSolverParameterPointers(const TerraSolverParameters& p, const std::vector<void*>& images) {
        for (int i = 0; i < 36; ++i) {
            floatPointers[i] = ((float*)(&p)) + i;
        }
        for (int i = 0; i < 3; ++i) {
            uintPointers[i] = (unsigned int*)(floatPointers[35] + 1) + i;
        }
        for (int i = 0; i < 6; ++i) {
            imagePointers[i] = images[i];
        }
    }
};