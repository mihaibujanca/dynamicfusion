#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include "../../shared/cudaUtil.h"
#include "Configure.h"
#include "CUDAWarpingSolver.h"
#include "OpenMesh.h"
#include "CeresSolver.h"
#include "../../shared/SolverIteration.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"
#include "../../shared/OptGraph.h"
#include <cuda_profiler_api.h>

class CombinedSolver : public CombinedSolverBase
{
	public:
        CombinedSolver(const SimpleMesh* mesh, std::vector<int> constraintsIdx, std::vector<std::vector<float>> constraintsTarget, CombinedSolverParameters params, float weightFit, float weightReg) :
            m_constraintsIdx(constraintsIdx), m_constraintsTarget(constraintsTarget)
		{
            m_weightFitSqrt = sqrtf(weightFit);
            m_weightRegSqrt = sqrtf(weightReg);
			m_result = *mesh;
			m_initial = m_result;
            m_combinedSolverParameters = params;

			unsigned int N = (unsigned int)mesh->n_vertices();

            m_dims = { N };
            m_vertexPosFloat3           = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
            m_anglesFloat3              = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
            m_vertexPosFloat3Urshape    = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
            m_vertexPosTargetFloat3     = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

            initializeConnectivity();
			resetGPUMemory();
            
            addSolver(std::make_shared<CUDAWarpingSolver>(N, d_numNeighbours, d_neighbourIdx, d_neighbourOffset), "CUDA", m_combinedSolverParameters.useCUDA);
            addSolver(std::make_shared<CeresSolver>(m_dims, &m_initial), "Ceres", m_combinedSolverParameters.useCeres);
            addOptSolvers(m_dims, "arap_mesh_deformation.t", m_combinedSolverParameters.optDoublePrecision);
		} 

        virtual void combinedSolveInit() override {

            m_problemParams.set("w_fitSqrt", &m_weightFitSqrt);
            m_problemParams.set("w_regSqrt", &m_weightRegSqrt);
            m_problemParams.set("Offset", m_vertexPosFloat3);
            m_problemParams.set("Angle", m_anglesFloat3);
            m_problemParams.set("UrShape", m_vertexPosFloat3Urshape);
            m_problemParams.set("Constraints", m_vertexPosTargetFloat3);
            m_problemParams.set("G", m_graph);

            m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
            m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
        }

        virtual void preNonlinearSolve(int i) override {
            setConstraints((float)(i+1) / (float)(m_combinedSolverParameters.numIter));
        }
        virtual void postNonlinearSolve(int) override{}

        virtual void preSingleSolve() override {
            m_result = m_initial;
            resetGPUMemory();
        }
        virtual void postSingleSolve() override {
            copyResultToCPUFromFloat3();
        }
        virtual void combinedSolveFinalize() override {
            if (m_combinedSolverParameters.profileSolve) {
                ceresIterationComparison("ARAP Mesh Deformation", m_combinedSolverParameters.optDoublePrecision);
            }
        }

		void setConstraints(float alpha)
		{
			unsigned int N = (unsigned int)m_result.n_vertices();
			std::vector<float3> h_vertexPosTargetFloat3(N);
			for (unsigned int i = 0; i < N; i++)
			{
				h_vertexPosTargetFloat3[i] = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
			}

			for (unsigned int i = 0; i < m_constraintsIdx.size(); i++)
			{
				const Vec3f& pt = m_result.point(VertexHandle(m_constraintsIdx[i]));
				const Vec3f target = Vec3f(m_constraintsTarget[i][0], m_constraintsTarget[i][1], m_constraintsTarget[i][2]);

				Vec3f z = (1 - alpha)*pt + alpha*target;
				h_vertexPosTargetFloat3[m_constraintsIdx[i]] = make_float3(z[0], z[1], z[2]);
			}
            m_vertexPosTargetFloat3->update(h_vertexPosTargetFloat3);
		}

        void initializeConnectivity() {
            unsigned int N = (unsigned int)m_initial.n_vertices();
            unsigned int E = (unsigned int)m_initial.n_edges();
            cudaSafeCall(cudaMalloc(&d_numNeighbours, sizeof(int)*N));
            cudaSafeCall(cudaMalloc(&d_neighbourIdx, sizeof(int) * 2 * E));
            cudaSafeCall(cudaMalloc(&d_neighbourOffset, sizeof(int)*(N + 1)));

            std::vector<int>	h_numNeighbours(N);
            std::vector<int>	h_neighbourIdx(2 * E);
            std::vector<int>	h_neighbourOffset(N + 1);

            unsigned int count = 0;
            unsigned int offset = 0;
            h_neighbourOffset[0] = 0;
            for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
            {
                VertexHandle c_vh(*v_it);
                unsigned int valance = m_initial.valence(c_vh);
                h_numNeighbours[count] = valance;

                for (SimpleMesh::VertexVertexIter vv_it = m_initial.vv_iter(c_vh); vv_it.is_valid(); vv_it++)
                {
                    VertexHandle v_vh(*vv_it);

                    h_neighbourIdx[offset] = v_vh.idx();
                    offset++;
                }

                h_neighbourOffset[count + 1] = offset;

                count++;
            }
            m_graph = createGraphFromNeighborLists(h_neighbourIdx, h_neighbourOffset);



            cudaSafeCall(cudaMemcpy(d_numNeighbours, h_numNeighbours.data(), sizeof(int)*N, cudaMemcpyHostToDevice));
            cudaSafeCall(cudaMemcpy(d_neighbourIdx, h_neighbourIdx.data(), sizeof(int) * 2 * E, cudaMemcpyHostToDevice));
            cudaSafeCall(cudaMemcpy(d_neighbourOffset, h_neighbourOffset.data(), sizeof(int)*(N + 1), cudaMemcpyHostToDevice));
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
			
			// Constraints
			setConstraints(1.0f);

			// Angles
			std::vector<float3> h_angles(N);
			for (unsigned int i = 0; i < N; i++)
			{
				h_angles[i] = make_float3(0.0f, 0.0f, 0.0f);
			}
            
            m_anglesFloat3->update(h_angles);
            m_vertexPosFloat3->update(h_vertexPosFloat3);
            m_vertexPosFloat3Urshape->update(h_vertexPosFloat3);
		}

        ~CombinedSolver()
		{
			cudaSafeCall(cudaFree(d_numNeighbours));
			cudaSafeCall(cudaFree(d_neighbourIdx));
			cudaSafeCall(cudaFree(d_neighbourOffset));
		}

        SimpleMesh* result() {
            return &m_result;
        }

		void copyResultToCPUFromFloat3()
		{
			unsigned int N = (unsigned int)m_result.n_vertices();
			std::vector<float3> h_vertexPosFloat3(N);
            m_vertexPosFloat3->copyTo(h_vertexPosFloat3);
			for (unsigned int i = 0; i < N; i++)
			{
				m_result.set_point(VertexHandle(i), Vec3f(h_vertexPosFloat3[i].x, h_vertexPosFloat3[i].y, h_vertexPosFloat3[i].z));
			}
		}

	private:

		SimpleMesh m_result;
		SimpleMesh m_initial;

        float m_weightFitSqrt;
        float m_weightRegSqrt;

        std::vector<unsigned int> m_dims;
	
        std::shared_ptr<OptImage> m_vertexPosFloat3Urshape;
        std::shared_ptr<OptImage> m_vertexPosFloat3;
        std::shared_ptr<OptImage> m_vertexPosTargetFloat3;
        std::shared_ptr<OptImage> m_anglesFloat3;
        std::shared_ptr<OptGraph> m_graph;

		int*	d_numNeighbours;
		int*	d_neighbourIdx;
		int* 	d_neighbourOffset;

		std::vector<int>				m_constraintsIdx;
		std::vector<std::vector<float>>	m_constraintsTarget;
};
