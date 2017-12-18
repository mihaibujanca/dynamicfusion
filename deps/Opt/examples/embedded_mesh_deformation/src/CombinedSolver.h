#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>

#include "OpenMesh.h"
#include "../../shared/SolverIteration.h"
#include "../../shared/CombinedSolverBase.h"
#include "../../shared/CombinedSolverParameters.h"

class CombinedSolver : public CombinedSolverBase
{
	public:
        CombinedSolver(const SimpleMesh* mesh, std::vector<int> constraintsIdx, std::vector<std::vector<float>> constraintsTarget, CombinedSolverParameters params) : m_constraintsIdx(constraintsIdx), m_constraintsTarget(constraintsTarget)
		{
            m_combinedSolverParameters = params;
			m_result = *mesh;
			m_initial = m_result;

            unsigned int N = (unsigned int)mesh->n_vertices();
            initializeConnectivity();
            std::vector<unsigned int> dims = { N };
            m_rotationMatrices = createEmptyOptImage(dims, OptImage::Type::FLOAT, 9, OptImage::GPU, true);
            m_vertexPositions = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
            m_vertexTargets = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
            m_urshape = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
            addOptSolvers(dims, "embedded_mesh_deformation.t", m_combinedSolverParameters.optDoublePrecision);
		} 

        virtual void combinedSolveInit() override {

            float weightFit = 3.0f;
            float weightReg = 12.0f;
            float weightRot = 5.0f;
            m_weightFitSqrt = sqrtf(weightFit);
            m_weightRegSqrt = sqrtf(weightReg);
            m_weightRotSqrt = sqrtf(weightRot);

            m_problemParams.set("w_fitSqrt", &m_weightFitSqrt);
            m_problemParams.set("w_regSqrt", &m_weightRegSqrt);
            m_problemParams.set("w_rotSqrt", &m_weightRotSqrt);
            m_problemParams.set("Offset", m_vertexPositions);
            m_problemParams.set("RotMatrix", m_rotationMatrices);
            m_problemParams.set("UrShape", m_urshape);
            m_problemParams.set("Constraints", m_vertexTargets);
            m_problemParams.set("G", m_graph);


            m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
            m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
        }

        virtual void preNonlinearSolve(int i) override {
            printf("preNonlinearSolve\n");
            setConstraints((float)(i+1) / (float)(m_combinedSolverParameters.numIter));
        }
        virtual void postNonlinearSolve(int) override{}
        
        virtual void preSingleSolve() override {
            printf("preSingleSolve\n");
            m_result = m_initial;
            resetGPUMemory();
        }
        virtual void postSingleSolve() override {
            copyResultToCPUFromFloat3();
        }
        virtual void combinedSolveFinalize() override {
            reportFinalCosts("Embedded Mesh Deformation", m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"), nan(""));
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
            m_vertexTargets->update(h_vertexPosTargetFloat3);
		}

        void initializeConnectivity() {
            unsigned int N = (unsigned int)m_initial.n_vertices();
            unsigned int E = (unsigned int)m_initial.n_edges();
            std::vector<int>    h_numNeighbours(N);
            std::vector<int>    h_neighbourIdx(2 * E);
            std::vector<int>    h_neighbourOffset(N + 1);
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
        }

		void resetGPUMemory()
		{
			unsigned int N = (unsigned int)m_initial.n_vertices();
			std::vector<float3> h_vertexPosFloat3(N);
			for (unsigned int i = 0; i < N; i++)
			{
				const Vec3f& pt = m_initial.point(VertexHandle(i));
				h_vertexPosFloat3[i] = make_float3(pt[0], pt[1], pt[2]);
			}
            
			// Constraints
			setConstraints(1.0f);

			// Angles
			std::vector<ml::mat3f> h_rots(N);
			for (unsigned int i = 0; i < N; i++)
			{
				h_rots[i].setIdentity();
			}
            m_rotationMatrices->update(h_rots);
            m_vertexPositions->update(h_vertexPosFloat3);
            m_urshape->update(h_vertexPosFloat3);
		}

		void copyResultToCPUFromFloat3()
		{
			unsigned int N = (unsigned int)m_result.n_vertices();
			std::vector<float3> h_vertexPosFloat3(N);
            m_vertexPositions->copyTo(h_vertexPosFloat3);

			for (unsigned int i = 0; i < N; i++)
			{
				m_result.set_point(VertexHandle(i), Vec3f(h_vertexPosFloat3[i].x, h_vertexPosFloat3[i].y, h_vertexPosFloat3[i].z));
			}
		}

        SimpleMesh* result() {
            return &m_result;
        }

	private:

		SimpleMesh m_result;
		SimpleMesh m_initial;
        float m_weightFitSqrt;
        float m_weightRegSqrt;
        float m_weightRotSqrt;
	
		std::shared_ptr<OptImage>   m_rotationMatrices;
        std::shared_ptr<OptImage>   m_vertexTargets;
        std::shared_ptr<OptImage>   m_vertexPositions;
        std::shared_ptr<OptImage>   m_urshape;
        std::shared_ptr<OptGraph>   m_graph;

		std::vector<int>				m_constraintsIdx;
		std::vector<std::vector<float>>	m_constraintsTarget;
};
