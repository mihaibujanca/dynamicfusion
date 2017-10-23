#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cudaUtil.h>
#include "OpenMesh.h"
#include <SolverIteration.h>
#include <CombinedSolverParameters.h>
#include <CombinedSolverBase.h>
#include <OptGraph.h>
#include <cuda_profiler_api.h>

#define MAX_K 20

static float clamp(float v, float mn, float mx) {
    return std::max(mn,std::min(v, mx));
}

class CombinedSolver : public CombinedSolverBase
{

public:
    CombinedSolver(kfusion::WarpField *warpField,
                   const std::vector<cv::Vec3f> &canonical_vertices,
                   const std::vector<cv::Vec3f> &canonical_normals,
                   const std::vector<cv::Vec3f> &live_vertices,
                   const std::vector<cv::Vec3f> &live_normals,
                   CombinedSolverParameters params)
    {
        warp = warpField;

        unsigned int D = warp->getNodes()->size();
        unsigned int N = canonical_vertices.size();

        m_dims = { D, N };

        m_rotationDeform    = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
        m_translationDeform = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

        m_canonicalVertices = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
        m_liveVertices      = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

        m_canonicalNormals  = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
        m_liveNormals       = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

        m_weights           = createEmptyOptImage({N}, OptImage::Type::FLOAT, KNN_NEIGHBOURS, OptImage::GPU, true);

        initializeConnectivity(canonical_vertices);
        resetGPUMemory();

        addOptSolvers(m_dims, "/home/mihai/Projects/dynamicfusion/tests/opt/dynamicfusion.t", m_combinedSolverParameters.optDoublePrecision); //FIXME: remove hardcoded path
    }


    void initializeConnectivity(const std::vector<cv::Vec3f> canonical_vertices)
    {
        unsigned int N = canonical_vertices.size();
        int count = 0;

        std::vector<std::vector<int> > graph_vector(KNN_NEIGHBOURS + 1, vector<int>(N));
        std::vector<float> weights(N * KNN_NEIGHBOURS);

        for(auto vertex : canonical_vertices)
        {
            graph_vector[0].push_back(count);
            warp->getWeightsAndUpdateKNN(vertex, &weights[count * KNN_NEIGHBOURS]);
            for(int i = 1; i < graph_vector.size(); i++)
                graph_vector[i].push_back((int)warp->getRetIndex()->at(i-1));
            count++;
        }
        m_weights->update(weights);
        m_data_graph = std::make_shared<OptGraph>(graph_vector);

    }

    virtual void combinedSolveInit() override
    {
        m_functionTolerance = 0.0000001f;

        m_problemParams.set("RotationDeform", m_rotationDeform);
        m_problemParams.set("TranslationDeform", m_translationDeform);

        m_problemParams.set("CanonicalVertices", m_canonicalVertices);
        m_problemParams.set("LiveVertices", m_liveVertices);

        m_problemParams.set("CanonicalNormals", m_canonicalNormals);
        m_problemParams.set("LiveNormals", m_liveNormals);

        m_problemParams.set("Weights", m_weights);

        m_problemParams.set("DataG", m_data_graph);
//        m_problemParams.set("RegG", m_reg_graph);

        m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
        m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
        m_solverParams.set("function_tolerance", &m_functionTolerance);
    }
    virtual void preSingleSolve() override {
        resetGPUMemory();
    }
    virtual void postSingleSolve() override {
        copyResultToCPUFromFloat3();
    }

    virtual void preNonlinearSolve(int) override {}

    virtual void postNonlinearSolve(int) override {}

    virtual void combinedSolveFinalize() override {
        reportFinalCosts("Robust Mesh Deformation", m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"), nan(""));
    }

    void resetGPUMemory()
    {
        uint N = (uint)warp->getNodes()->size();
        std::vector<float3> h_translation(N);
        std::vector<float3> h_rotation(N);

        for(int i = 0; i < N; i++)
        {
            float x,y,z;
            warp->getNodes()->at(i).transform.getTranslation(x,y,z);
            h_translation[i] = make_float3(x,y,z);

            warp->getNodes()->at(i).transform.getRotation().getRodrigues(x,y,z);
            h_rotation[i] = make_float3(x,y,z);
        }

        m_rotationDeform->update(h_rotation);
        m_translationDeform->update(h_translation);//TODO: use the node transformations themselves instead.
    }

    std::vector<cv::Vec3f> result()
    {
        return std::vector<cv::Vec3f>();
    }

    void copyResultToCPUFromFloat3()
    {
        unsigned int N = (unsigned int)warp->getNodes()->size();
        std::vector<float3> h_translation(N);
        m_translationDeform->copyTo(h_translation);

        for (unsigned int i = 0; i < N; i++)
            warp->getNodes()->at(i).transform.encodeTranslation(h_translation[i].x, h_translation[i].y, h_translation[i].z);
    }

private:

    kfusion::WarpField *warp;

    ml::Timer m_timer;

    // Current index in solve
    std::vector<unsigned int> m_dims;

    std::shared_ptr<OptImage> m_rotationDeform;
    std::shared_ptr<OptImage> m_translationDeform;

    std::shared_ptr<OptImage> m_canonicalVertices;
    std::shared_ptr<OptImage> m_liveVertices;
    std::shared_ptr<OptImage> m_canonicalNormals;
    std::shared_ptr<OptImage> m_liveNormals;
    std::shared_ptr<OptImage> m_weights;
    std::shared_ptr<OptGraph> m_reg_graph;
    std::shared_ptr<OptGraph> m_data_graph;
    float m_functionTolerance;
};

