#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>

#include "CUDAWarpingSolver.h"
#include "CeresSolver.h"
#include "OpenMesh.h"

#include "../../shared/CombinedSolverBase.h"

class CombinedSolver : public CombinedSolverBase
{
	public:
        CombinedSolver(const SimpleMesh* mesh, int3 voxelGridSize, CombinedSolverParameters params)
		{
            m_combinedSolverParameters = params;

			m_result = *mesh;
			m_initial = m_result;

            m_dims      = voxelGridSize;
			m_nNodes    = (m_dims.x + 1)*(m_dims.y + 1)*(m_dims.z + 1);
			
			unsigned int N = (unsigned int)mesh->n_vertices();
		
            std::vector<unsigned int> dims = { (uint)m_dims.x + 1, (uint)m_dims.y + 1, (uint)m_dims.z + 1 };
            m_gridPosFloat3         = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
            m_gridAnglesFloat3      = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
            m_gridPosFloat3Urshape  = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
            m_gridPosTargetFloat3   = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

			m_vertexToVoxels.resize(N);
            m_relativeCoords.resize(N);

			resetGPUMemory();
            
            addSolver(std::make_shared<CUDAWarpingSolver>(dims), "CUDA", m_combinedSolverParameters.useCUDA);
            addSolver(std::make_shared<CeresSolver>(dims), "Ceres", m_combinedSolverParameters.useCeres);
            addOptSolvers(dims, "volumetric_mesh_deformation.t", m_combinedSolverParameters.optDoublePrecision);
		}

        virtual void combinedSolveInit() override {
            float weightFit = 1.0f;
            float weightReg = 0.05f;

            m_weightFitSqrt = sqrtf(weightFit);
            m_weightRegSqrt = sqrtf(weightReg);

            m_problemParams.set("Offset", m_gridPosFloat3);
            m_problemParams.set("Angle", m_gridAnglesFloat3);
            m_problemParams.set("UrShape", m_gridPosFloat3Urshape);
            m_problemParams.set("Constraints", m_gridPosTargetFloat3);
            m_problemParams.set("w_fitSqrt", &m_weightFitSqrt);
            m_problemParams.set("w_regSqrt", &m_weightRegSqrt);

            
            m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
            m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
        }
        virtual void preSingleSolve() override {
            m_result = m_initial;
            resetGPUMemory();
        }
        virtual void postSingleSolve() override {
            copyResultToCPUFromFloat3();
        }

        virtual void preNonlinearSolve(int) override {}
        virtual void postNonlinearSolve(int) override{}

        virtual void combinedSolveFinalize() override {
            if (m_combinedSolverParameters.profileSolve) {
                ceresIterationComparison("Volumetric Mesh Deformation", m_combinedSolverParameters.optDoublePrecision);
            }
        }

		void setConstraints(float alpha)
		{
			std::vector<float3> h_gridPosTargetFloat3(m_nNodes);
			for (int i = 0; i <= m_dims.x; i++)
			{
				for (int j = 0; j <= m_dims.y; j++)
				{
					for (int k = 0; k <= m_dims.z; k++)
					{
						int index = getIndex1D(make_int3(i, j, k));
						vec3f delta(m_delta.x, m_delta.y, m_delta.z);
						mat3f fac = mat3f::diag((float)i, (float)j, (float)k);
						vec3f min(m_min.x, m_min.y, m_min.z);
						vec3f v = min + fac*delta;

						if (j == 0) { h_gridPosTargetFloat3[index] = make_float3(v.x, v.y, v.z); }
						else if (j == m_dims.y)	{
							mat3f f = mat3f::diag(m_dims.x / 2.0f, (float)m_dims.y, m_dims.z / 2.0f);
							vec3f mid = vec3f(m_min.x, m_min.y, m_min.z) + f*delta; mat3f R = ml::mat3f::rotationZ(-90.0f);
							v = R*(v - mid) + mid + vec3f(2.5f, -2.5f, 0.0f);
							h_gridPosTargetFloat3[index] = make_float3(v.x, v.y, v.z);
						}
						else { h_gridPosTargetFloat3[index] = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()); }
					}
				}
			}
            m_gridPosTargetFloat3->update(h_gridPosTargetFloat3);
		}

		void computeBoundingBox()
		{
			m_min = make_float3(+std::numeric_limits<float>::max(), +std::numeric_limits<float>::max(), +std::numeric_limits<float>::max());
			m_max = make_float3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
			for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
			{
				SimpleMesh::Point p = m_initial.point(VertexHandle(*v_it));
				m_min.x = fmin(m_min.x, p[0]); m_min.y = fmin(m_min.y, p[1]); m_min.z = fmin(m_min.z, p[2]);
				m_max.x = fmax(m_max.x, p[0]); m_max.y = fmax(m_max.y, p[1]); m_max.z = fmax(m_max.z, p[2]);
			}
		}

		void GraphAddSphere(SimpleMesh& out, SimpleMesh::Point offset, float scale, SimpleMesh meshSphere, bool constrained)
		{
			unsigned int currentN = (unsigned int)out.n_vertices();
			for (unsigned int i = 0; i < meshSphere.n_vertices(); i++)
			{
				SimpleMesh::Point p = meshSphere.point(VertexHandle(i))*scale + offset;
				VertexHandle vh = out.add_vertex(p);
				out.set_color(vh, SimpleMesh::Color(200, 0, 0));
			}

			for (unsigned int k = 0; k < meshSphere.n_faces(); k++)
			{
				std::vector<VertexHandle> vhs;
				for (SimpleMesh::FaceVertexIter v_it = meshSphere.fv_begin(FaceHandle(k)); v_it != meshSphere.fv_end(FaceHandle(k)); ++v_it)
				{
					vhs.push_back(VertexHandle(v_it->idx() + currentN));
				}
				out.add_face(vhs);
			}
		}

		void GraphAddCone(SimpleMesh& out, SimpleMesh::Point offset, float scale, float3 direction, SimpleMesh meshCone)
		{
			unsigned int currentN = (unsigned int)out.n_vertices();
			for (unsigned int i = 0; i < meshCone.n_vertices(); i++)
			{
				SimpleMesh::Point pO = meshCone.point(VertexHandle(i));

				vec3f o(0.0f, 0.5f, 0.0f);
				mat3f s = ml::mat3f::diag(scale, length(direction), scale);
				vec3f p(pO[0], pO[1], pO[2]);

				p = s*(p + o);

				vec3f f(direction.x, direction.y, direction.z); f = ml::vec3f::normalize(f);
				vec3f up(0.0f, 1.0f, 0.0f);
				vec3f axis  = ml::vec3f::cross(up, f);
				float angle = acos(ml::vec3f::dot(up, f))*180.0f/(float)PI;
				mat3f R = ml::mat3f::rotation(axis, angle); if (axis.length() < 0.00001f) R = ml::mat3f::identity();
				
				p = R*p;

				VertexHandle vh = out.add_vertex(SimpleMesh::Point(p.x, p.y, p.z) + offset);
				out.set_color(vh, SimpleMesh::Color(70, 200, 70));
			}

			for (unsigned int k = 0; k < meshCone.n_faces(); k++)
			{
				std::vector<VertexHandle> vhs;
				for (SimpleMesh::FaceVertexIter v_it = meshCone.fv_begin(FaceHandle(k)); v_it != meshCone.fv_end(FaceHandle(k)); ++v_it)
				{
					vhs.push_back(VertexHandle(v_it->idx() + currentN));
				}
				out.add_face(vhs);
			}
		}

		void saveGraph(const std::string& filename, float3* data, unsigned int N, float scale, SimpleMesh meshSphere, SimpleMesh meshCone)
		{
			SimpleMesh out;

			std::vector<float3> h_gridPosTarget(m_nNodes);
            m_gridPosTargetFloat3->copyTo(h_gridPosTarget);
			for (unsigned int i = 0; i < N; i++)
			{
				if (h_gridPosTarget[i].x != -std::numeric_limits<float>::infinity())
				{
					GraphAddSphere(out, SimpleMesh::Point(data[i].x, data[i].y, data[i].z), scale, meshSphere, h_gridPosTarget[i].x != -std::numeric_limits<float>::infinity());
				}
			}

			for (int i = 0; i <= m_dims.x; i++)
			{
				for (int j = 0; j <= m_dims.y; j++)
				{
					for (int k = 0; k <= m_dims.z; k++)
					{
						float3 pos0 = data[getIndex1D(make_int3(i, j, k))];
						
						if (i + 1 <= m_dims.x) { float3 dir1 = data[getIndex1D(make_int3(i + 1, j, k))] - pos0; GraphAddCone(out, SimpleMesh::Point(pos0.x, pos0.y, pos0.z), scale*0.25f, dir1, meshCone); }
						if (j + 1 <= m_dims.y) { float3 dir2 = data[getIndex1D(make_int3(i, j + 1, k))] - pos0; GraphAddCone(out, SimpleMesh::Point(pos0.x, pos0.y, pos0.z), scale*0.25f, dir2, meshCone); }
						if (k + 1 <= m_dims.z) { float3 dir3 = data[getIndex1D(make_int3(i, j, k + 1))] - pos0; GraphAddCone(out, SimpleMesh::Point(pos0.x, pos0.y, pos0.z), scale*0.25f, dir3, meshCone); }
					}
				}
			}

			OpenMesh::IO::write_mesh(out, filename, IO::Options::VertexColor);
		}

		void initializeWarpGrid()
		{
			std::vector<float3> h_gridVertexPosFloat3(m_nNodes);
			for (int i = 0; i <= m_dims.x; i++)
			{
				for (int j = 0; j <= m_dims.y; j++)
				{
					for (int k = 0; k <= m_dims.z; k++)
					{
						float3 fac = make_float3((float)i, (float)j, (float)k);
						float3 v = m_min + fac*m_delta;
						h_gridVertexPosFloat3[getIndex1D(make_int3(i, j, k))] = v;
					}
				}
			}

            m_gridPosFloat3->update(h_gridVertexPosFloat3);
            m_gridPosFloat3Urshape->update(h_gridVertexPosFloat3);
			cudaSafeCall(cudaMemset(m_gridAnglesFloat3->data(), 0, sizeof(float3)*m_nNodes));
		}

		void resetGPUMemory()
		{
			computeBoundingBox();

			float EPS = 0.000001f;
			m_min -= make_float3(EPS, EPS, EPS);
			m_max += make_float3(EPS, EPS, EPS);
			m_delta = (m_max - m_min); m_delta.x /= (m_dims.x); m_delta.y /= (m_dims.y); m_delta.z /= (m_dims.z);

			initializeWarpGrid();

			for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
			{
			    VertexHandle c_vh(*v_it);
				SimpleMesh::Point p = m_initial.point(c_vh);
				float3 pp = make_float3(p[0], p[1], p[2]);

				pp = (pp - m_min);
				pp.x /= m_delta.x;
				pp.y /= m_delta.y;
				pp.z /= m_delta.z;

				int3 pInt = make_int3((int)pp.x, (int)pp.y, (int)pp.z);
				m_vertexToVoxels[c_vh.idx()] = pInt;
				m_relativeCoords[c_vh.idx()] = pp - make_float3((float)pInt.x, (float)pInt.y, (float)pInt.z);
			}

			// Constraints
			setConstraints(1.0f);
		}


        SimpleMesh* result() {
            return &m_result;
        }

		int getIndex1D(int3 idx)
        {
			return idx.x*((m_dims.y + 1)*(m_dims.z + 1)) + idx.y*(m_dims.z + 1) + idx.z;
		}

		void saveGraphResults()
		{
			SimpleMesh meshSphere;
			if (!OpenMesh::IO::read_mesh(meshSphere, "../data/sphere.ply"))
			{
				std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
				exit(1);
			}

			SimpleMesh meshCone;
			if (!OpenMesh::IO::read_mesh(meshCone, "../data/cone.ply"))
			{
				std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
				exit(1);
			}

			std::vector<float3> h_gridPosUrshapeFloat3(m_nNodes);
            m_gridPosFloat3Urshape->copyTo(h_gridPosUrshapeFloat3);   saveGraph("grid.ply", h_gridPosUrshapeFloat3.data(), m_nNodes, 0.05f, meshSphere, meshCone);

            std::vector<float3> h_gridPosFloat3(m_nNodes);
            m_gridPosFloat3->copyTo(h_gridPosFloat3);
			saveGraph("gridOut.ply", h_gridPosFloat3.data(), m_nNodes, 0.05f, meshSphere, meshCone);
		}

		void copyResultToCPUFromFloat3()
		{
            std::vector<float3> h_gridPosFloat3(m_nNodes);
            m_gridPosFloat3->copyTo(h_gridPosFloat3);

			for (SimpleMesh::VertexIter v_it = m_result.vertices_begin(); v_it != m_result.vertices_end(); ++v_it)
			{
				VertexHandle vh(*v_it);

				int3   voxelId = m_vertexToVoxels[vh.idx()];
				float3 relativeCoords = m_relativeCoords[vh.idx()];

				float3 p000 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 0, 0))];
				float3 p001 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 0, 1))];
				float3 p010 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 1, 0))];
				float3 p011 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 1, 1))];
				float3 p100 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 0, 0))];
				float3 p101 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 0, 1))];
				float3 p110 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 1, 0))];
				float3 p111 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 1, 1))];

				float3 px00 = (1.0f - relativeCoords.x)*p000 + relativeCoords.x*p100;
				float3 px01 = (1.0f - relativeCoords.x)*p001 + relativeCoords.x*p101;
				float3 px10 = (1.0f - relativeCoords.x)*p010 + relativeCoords.x*p110;
				float3 px11 = (1.0f - relativeCoords.x)*p011 + relativeCoords.x*p111;

				float3 pxx0 = (1.0f - relativeCoords.y)*px00 + relativeCoords.y*px10;
				float3 pxx1 = (1.0f - relativeCoords.y)*px01 + relativeCoords.y*px11;

				float3 p = (1.0f - relativeCoords.z)*pxx0 + relativeCoords.z*pxx1;

				m_result.set_point(vh, SimpleMesh::Point(p.x, p.y, p.z));
			}
		}

	private:

		SimpleMesh m_result;
		SimpleMesh m_initial;

		int	   m_nNodes;

		float3 m_min;
		float3 m_max;

		int3   m_dims;
		float3 m_delta;

		std::vector<int3>   m_vertexToVoxels;
        std::vector<float3> m_relativeCoords;

        std::shared_ptr<OptImage> m_gridPosTargetFloat3;
        std::shared_ptr<OptImage> m_gridPosFloat3;
        std::shared_ptr<OptImage> m_gridPosFloat3Urshape;
        std::shared_ptr<OptImage> m_gridAnglesFloat3;
        float m_weightFitSqrt;
        float m_weightRegSqrt;
};
