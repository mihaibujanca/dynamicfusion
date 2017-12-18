#pragma once

#include "../../shared/OptUtils.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"
#include "mLibInclude.h"

#include "../../shared/cudaUtil.h"

#include "OpenMesh.h"


class CombinedSolver : public CombinedSolverBase
{
public:
    CombinedSolver(const SimpleMesh* mesh, bool performanceRun, CombinedSolverParameters params, float weightFit, float weightReg)
    {
        m_weightFitSqrt = sqrtf(weightFit);
        m_weightRegSqrt = sqrtf(weightReg);
        m_combinedSolverParameters = params;
        m_result = *mesh;
        m_initial = m_result;

        unsigned int N = (unsigned int)mesh->n_vertices();
        initializeConnectivity();
        std::vector<unsigned int> dims = { N };
        m_unknown = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
        m_target = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

        std::cout << "compiling... ";
        addOptSolvers(dims, "cotangent_mesh_smoothing.t", m_combinedSolverParameters.optDoublePrecision);
        std::cout << " done!" << std::endl;
    }

    virtual void combinedSolveInit() override {

        m_problemParams.set("w_fitSqrt", &m_weightFitSqrt);
        m_problemParams.set("w_regSqrt", &m_weightRegSqrt);
        m_problemParams.set("X", m_unknown);
        m_problemParams.set("A", m_target);
        m_problemParams.set("G", m_graph);

        m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
        m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
    }

    virtual void preNonlinearSolve(int) override {}
    virtual void postNonlinearSolve(int) override{}

    virtual void preSingleSolve() override {
        m_result = m_initial;
        resetGPUMemory();
    }
    virtual void postSingleSolve() override {
        copyResultToCPUFromFloat3();
    }
    virtual void combinedSolveFinalize() override {
        reportFinalCosts("Cotangent Mesh Smoothing", m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"), nan(""));
    }

    void initializeConnectivity() {
        bool isTri = m_initial.is_triangles();
        if (!isTri) {
            std::cout << "MUST BE A TRI MESH" << std::endl;
            exit(1);
        }

        unsigned int N = (unsigned int)m_initial.n_vertices();
        unsigned int E = (unsigned int)m_initial.n_edges();

        std::vector<int>	h_neighbourIdx(2 * E * 3);
        std::vector<int>	h_neighbourOffset(N + 1);
        unsigned int count = 0;
        unsigned int offset = 0;
        h_neighbourOffset[0] = 0;
        for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
        {
            VertexHandle c_vh(*v_it);
            std::vector<unsigned int> neighborIndices;
            for (SimpleMesh::VertexVertexIter vv_it = m_initial.vv_iter(c_vh); vv_it.is_valid(); vv_it++) {
                VertexHandle v_vh(*vv_it);
                neighborIndices.push_back(v_vh.idx());
            }

            for (size_t i = 0; i < neighborIndices.size(); i++) {
                const unsigned int n = (unsigned int)neighborIndices.size();
                unsigned int prev = neighborIndices[(i + n - 1) % n];
                unsigned int next = neighborIndices[(i + 1) % n];
                unsigned int curr = neighborIndices[i];

                h_neighbourIdx[3 * offset + 0] = curr;
                h_neighbourIdx[3 * offset + 1] = prev;
                h_neighbourIdx[3 * offset + 2] = next;
                offset++;
            }

            h_neighbourOffset[count + 1] = offset;

            count++;
        }
        // Convert to Opt format
        std::vector<int> head;
        std::vector<int> tail;
        std::vector<int> prev;
        std::vector<int> next;
        for (int headX = 0; headX < (int)N; ++headX) {
            for (int j = h_neighbourOffset[headX]; j < h_neighbourOffset[headX + 1]; ++j) {
                head.push_back(headX);
                tail.push_back(h_neighbourIdx[3 * j + 0]);
                prev.push_back(h_neighbourIdx[3 * j + 1]);
                next.push_back(h_neighbourIdx[3 * j + 2]);
            }
        }

        m_graph = std::make_shared<OptGraph>(std::vector<std::vector<int> >({ head, tail, prev, next }));
    }

    void resetGPUMemory()
    {
        unsigned int N = (unsigned int)m_initial.n_vertices();
        unsigned int E = (unsigned int)m_initial.n_edges();
        std::vector<float3> h_vertexPosFloat3(N);
        for (unsigned int i = 0; i < N; i++)
        {
            const Vec3f& pt = m_initial.point(VertexHandle(i));
            h_vertexPosFloat3[i] = make_float3(pt[0], pt[1], pt[2]);
        }
        m_unknown->update(h_vertexPosFloat3);
        m_target->update(h_vertexPosFloat3);
    }

    void copyResultToCPUFromFloat3()
    {
        unsigned int N = (unsigned int)m_result.n_vertices();
        std::vector<float3> h_vertexPosFloat3(N);
        m_unknown->copyTo(h_vertexPosFloat3);

        for (unsigned int i = 0; i < N; i++)
        {
            m_result.set_point(VertexHandle(i), Vec3f(h_vertexPosFloat3[i].x, h_vertexPosFloat3[i].y, h_vertexPosFloat3[i].z));
        }
    }

    SimpleMesh* result() {
        return &m_result;
    }

private:

    float m_weightFitSqrt;
    float m_weightRegSqrt;

    std::shared_ptr<OptImage> m_unknown;
    std::shared_ptr<OptImage> m_target;
    std::shared_ptr<OptGraph> m_graph;

    SimpleMesh m_result;
    SimpleMesh m_initial;
};
