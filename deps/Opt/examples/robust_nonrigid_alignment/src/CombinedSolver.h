#pragma once
#include <nanoflann/include/nanoflann.hpp>
#include "mLibInclude.h"

#include <cuda_runtime.h>

#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"
#include "../../shared/SolverIteration.h"

#include "OpenMesh.h"

using namespace nanoflann;
// For nanoflann computation
struct PointCloud_nanoflann
{
    std::vector<float3>  pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline float kdtree_distance(const float *p1, const size_t idx_p2, size_t /*size*/) const
    {
        const float d0 = p1[0] - pts[idx_p2].x;
        const float d1 = p1[1] - pts[idx_p2].y;
        const float d2 = p1[2] - pts[idx_p2].z;
        return d0*d0 + d1*d1 + d2*d2;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, int dim) const
    {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};

typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<float, PointCloud_nanoflann>,
    PointCloud_nanoflann,
    3 /* dim */
> NanoKDTree;


static bool operator==(const float3& v0, const float3& v1) {
    return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
static bool operator!=(const float3& v0, const float3& v1) {
    return !(v0 == v1);
}
#define MAX_K 20

static float clamp(float v, float mn, float mx) {
    return std::max(mn,std::min(v, mx));
}

class CombinedSolver : public CombinedSolverBase
{

	public:
        CombinedSolver(const SimpleMesh* sourceMesh, const std::vector<SimpleMesh*>& targetMeshes, const std::vector<int4>& sourceTetIndices, CombinedSolverParameters params)
		{
            m_combinedSolverParameters = params;
            m_result = *sourceMesh;
			m_initial = m_result;
            m_sourceTetIndices = sourceTetIndices;

            for (SimpleMesh* mesh : targetMeshes) {
                m_targets.push_back(*mesh);
            }

            uint N = (uint)sourceMesh->n_vertices();
            uint E = (uint)sourceMesh->n_edges();
            
            double sumEdgeLength = 0.0f;
            for (auto edgeHandle : m_initial.edges()) {
                auto edge = m_initial.edge(edgeHandle);
                sumEdgeLength += m_initial.calc_edge_length(edgeHandle);
            }
            m_averageEdgeLength = sumEdgeLength / E;
            std::cout << "Average Edge Length: " << m_averageEdgeLength << std::endl;

            std::vector<unsigned int> dims = { N };
            m_vertexPosTargetFloat3     = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            m_vertexNormalTargetFloat3  = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            m_robustWeights             = createEmptyOptImage(dims, OptImage::Type::FLOAT, 1, OptImage::Location::GPU, true);
            m_vertexPosFloat3           = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            m_vertexPosFloat3Urshape    = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            m_anglesFloat3              = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            
			resetGPUMemory();   

            m_rnd       = std::mt19937(230948);

            float spuriousProbability = 0.05f;

            std::uniform_int_distribution<> targetDistribution(0, (int)m_targets[0].n_vertices() - 1);
            float noiseModifier = 30.0f;
            std::normal_distribution<> normalDistribution(0.0f, m_averageEdgeLength * noiseModifier);
            int spuriousCount = int(N*spuriousProbability);
            for (int i = 0; i < spuriousCount; ++i) {
                m_spuriousIndices.push_back(targetDistribution(m_rnd));
                m_noisyOffsets.push_back(make_float3((float)normalDistribution(m_rnd), (float)normalDistribution(m_rnd), (float)normalDistribution(m_rnd)));
            }

            if (!m_result.has_vertex_colors()) {
                m_result.request_vertex_colors();
            }

            for (unsigned int i = 0; i < N; i++)
            {
                uchar w = 255;
                m_result.set_color(VertexHandle(i), Vec3uc(w, w, w));
            }

            OpenMesh::IO::Options options = OpenMesh::IO::Options::VertexColor;
            int failure = OpenMesh::IO::write_mesh(m_result, "out_noisetemplate.ply", options);
            assert(failure);

            addOptSolvers(dims, "robust_nonrigid_alignment.t", m_combinedSolverParameters.optDoublePrecision);
		} 


        /** This solver is a bit more complicated than most of the other examples, since it continually solves slightly different optimization problems (different correspondences, targets)
            We'll just do a one-off override of the main solve function to handle this. */
        virtual void solveAll() override {
            combinedSolveInit();
            for (auto s : m_solverInfo) {
                if (s.enabled) {
                    m_result = m_initial;
                    resetGPUMemory();
                    m_problemParams.set("G", m_graph);
                    for (m_targetIndex = 0; m_targetIndex < m_targets.size(); ++m_targetIndex) {
                        singleSolve(s);
                    }
                }
            }
            combinedSolveFinalize();
        }

        virtual void combinedSolveInit() override {
            m_weightFit = 10.0f;
            m_weightRegMax = 64.0f;
            
            m_weightRegMin = 4.0f;
            m_weightRegFactor = 0.9f;

            m_weightReg = m_weightRegMax;

            m_functionTolerance = 0.0000001f;

            m_fitSqrt = sqrt(m_weightFit);
            m_regSqrt = sqrt(m_weightReg);

            m_problemParams.set("w_fit", &m_fitSqrt);
            m_problemParams.set("w_reg", &m_regSqrt);

            m_problemParams.set("Offset", m_vertexPosFloat3);
            m_problemParams.set("Angle", m_anglesFloat3);
            m_problemParams.set("RobustWeights", m_robustWeights);
            m_problemParams.set("UrShape", m_vertexPosFloat3Urshape);
            m_problemParams.set("Constraints", m_vertexPosTargetFloat3);
            m_problemParams.set("ConstraintNormals", m_vertexNormalTargetFloat3);
            m_problemParams.set("G", m_graph);

            m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
            m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
            m_solverParams.set("function_tolerance", &m_functionTolerance);
        }
        virtual void preSingleSolve() override {
            unsigned int N = (unsigned int)m_initial.n_vertices();
            m_timer.start();
            m_targetAccelerationStructure = generateAccelerationStructure(m_targets[m_targetIndex]);
            m_timer.stop();
            m_previousConstraints.resize(N);
            for (int i = 0; i < (int)N; ++i) {
                m_previousConstraints[i] = make_float3(0, 0, -90901283092183);
            }
            std::cout << "---- Acceleration Structure Build: " << m_timer.getElapsedTime() << "s" << std::endl;
            m_weightReg = m_weightRegMax;
        }
        virtual void postSingleSolve() override { 
            char buff[100];
            sprintf(buff, "out_%04d.ply", m_targetIndex);
            saveCurrentMesh(buff);
        }

        virtual void preNonlinearSolve(int) override {
            m_timer.start();
            int newConstraintCount = setConstraints(m_targetIndex, (float)m_averageEdgeLength*5.0f);
            m_timer.stop();
            double setConstraintsTime = m_timer.getElapsedTime();
            std::cout << "-- Set Constraints: " << setConstraintsTime << "s";
            std::cout << " -------- New constraints: " << newConstraintCount << std::endl;
            if (newConstraintCount <= 5) {
                std::cout << " -------- Few New Constraints" << std::endl;
                if (m_weightReg != m_weightRegMin) {
                    std::cout << " -------- Skipping to min reg weight" << std::endl;
                    m_weightReg = m_weightRegMin;
                }
                m_endSolveEarly = true;
            }
            m_regSqrt = sqrtf(m_weightReg);
        }
        virtual void postNonlinearSolve(int) override {
            m_timer.start();
            copyResultToCPUFromFloat3();
            m_timer.stop();
            double copyTime = m_timer.getElapsedTime();
            std::cout << "--Copy to CPU : " << copyTime << "s " << std::endl;
            m_weightReg = fmaxf(m_weightRegMin, m_weightReg*m_weightRegFactor);
        }

        virtual void combinedSolveFinalize() override {
            reportFinalCosts("Robust Mesh Deformation", m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"), nan(""));
        }


        inline vec3f convertDepthToRGB(float depth, float depthMin = 0.0f, float depthMax = 1.0f) {
            float depthZeroOne = (depth - depthMin) / (depthMax - depthMin);
            float x = 1.0f - depthZeroOne;
            if (x < 0.0f) x = 0.0f;
            if (x > 1.0f) x = 1.0f;
            return BaseImageHelper::convertHSVtoRGB(vec3f(240.0f*x, 1.0f, 0.5f));
        }

        void saveCurrentMesh(std::string filename) {
            { // Save intermediate mesh
                unsigned int N = (unsigned int)m_result.n_vertices();
                std::vector<float> h_vertexWeightFloat(N);
                m_robustWeights->copyTo(h_vertexWeightFloat);
                for (unsigned int i = 0; i < N; i++)
                {
                    vec3f color = convertDepthToRGB(1.0f-clamp(h_vertexWeightFloat[i], 0.0f, 1.0f));

                    m_result.set_color(VertexHandle(i), Vec3uc((uchar)(color.r * 255), (uchar)(color.g * 255), (uchar)(color.b * 255)));

                    if (h_vertexWeightFloat[i] < 0.9f || h_vertexWeightFloat[i] > 1.0f) {
                        printf("Interesting robustWeight[%d]: %f\n", i, h_vertexWeightFloat[i]);
                    }
                }
                
                OpenMesh::IO::Options options = OpenMesh::IO::Options::VertexColor;
                int failure = OpenMesh::IO::write_mesh(m_result, filename, options);
                assert(failure);
            }
        }


        int setConstraints(int targetIndex, float positionThreshold = std::numeric_limits<float>::infinity(), float cosNormalThreshold = 0.7f)
		{
            
			unsigned int N = (unsigned int)m_result.n_vertices();
			std::vector<float3> h_vertexPosTargetFloat3(N);
            std::vector<float3> h_vertexNormalTargetFloat3(N);

            if (!(m_targets[targetIndex].has_face_normals() && m_targets[targetIndex].has_vertex_normals())) { 
                m_targets[targetIndex].request_face_normals();
                m_targets[targetIndex].request_vertex_normals();
                // let the mesh update the normals
                m_targets[targetIndex].update_normals();
            }

            if (!(m_result.has_face_normals() && m_result.has_vertex_normals())) {
                m_result.request_face_normals();
                m_result.request_vertex_normals();
            }
            m_result.update_normals();

            uint thrownOutCorrespondenceCount = 0;
            float3 invalidPt = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());

//#pragma omp parallel for // TODO: check why it makes everything wrongs
            for (int i = 0; i < (int)N; i++) {
                std::vector<size_t> neighbors(MAX_K);
                std::vector<float> out_dist_sqr(MAX_K);
                auto currentPt = m_result.point(VertexHandle(i));
                auto sNormal = m_result.normal(VertexHandle(i));
                auto sourceNormal = make_float3(sNormal[0], sNormal[1], sNormal[2]);
                m_targetAccelerationStructure->knnSearch(currentPt.data(), MAX_K, &neighbors[0], &out_dist_sqr[0]);
                bool validTargetFound = false;
                for (size_t indexOfNearest : neighbors) {
                    const Vec3f target = m_targets[targetIndex].point(VertexHandle((int)indexOfNearest));
                    auto tNormal = m_targets[targetIndex].normal(VertexHandle((int)indexOfNearest));
                    auto targetNormal = make_float3(tNormal[0], tNormal[1], tNormal[2]);
                    float dist = (target - currentPt).length();
                    if (dist > positionThreshold) {
                        break;
                    }
                    if (dot(targetNormal, sourceNormal) > cosNormalThreshold) {
                        h_vertexPosTargetFloat3[i] = make_float3(target[0], target[1], target[2]);
                        h_vertexNormalTargetFloat3[i] = targetNormal;
                        validTargetFound = true;
                        break;
                    }
                    
                }
                if (!validTargetFound) {
                    ++thrownOutCorrespondenceCount;
                    h_vertexPosTargetFloat3[i] = invalidPt;
                }
			}

            for (int i = 0; i < m_spuriousIndices.size(); ++i) {
                h_vertexPosTargetFloat3[m_spuriousIndices[i]] += m_noisyOffsets[i];
            }


            m_vertexPosTargetFloat3->update(h_vertexPosTargetFloat3);
            m_vertexNormalTargetFloat3->update(h_vertexNormalTargetFloat3);

            std::vector<float>  h_robustWeights(N);
            m_robustWeights->copyTo(h_robustWeights);

            int constraintsUpdated = 0;
            for (int i = 0; i < (int)N; ++i) {
                if (m_previousConstraints[i] != h_vertexPosTargetFloat3[i]) {
                    m_previousConstraints[i] = h_vertexPosTargetFloat3[i];
                    ++constraintsUpdated;
                    {
                        auto currentPt = m_result.point(VertexHandle(i));
                        auto v = h_vertexPosTargetFloat3[i];
                        const Vec3f target = Vec3f(v.x, v.y, v.z);
                        float dist = (target - currentPt).length();
                        float weight = (positionThreshold - dist) / positionThreshold;
                        h_robustWeights[i] = fmaxf(0.1f, weight*0.9f+0.05f);
                        h_robustWeights[i] = 1.0f;
                    }
                }
            }
            m_robustWeights->update(h_robustWeights);

            std::cout << "*******Thrown out correspondence count: " << thrownOutCorrespondenceCount << std::endl;

            return constraintsUpdated;
		}

        void generateOptEdges(std::vector<int>& h_numNeighbours, std::vector<int>&	h_neighbourIdx, std::vector<int>& h_neighbourOffset) {
            uint N = (uint)m_initial.n_vertices();
            h_numNeighbours.resize(N);
            h_neighbourOffset.resize(N + 1);
            if (m_sourceTetIndices.size() == 0) {
                uint E = (uint)m_initial.n_edges();
                h_neighbourIdx.resize(2 * E);

                unsigned int count = 0;
                unsigned int offset = 0;
                h_neighbourOffset[0] = 0;
                for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
                {
                    VertexHandle c_vh(*v_it);
                    unsigned int valence = m_initial.valence(c_vh);
                    h_numNeighbours[count] = valence;
                    for (SimpleMesh::VertexVertexIter vv_it = m_initial.vv_iter(c_vh); vv_it.is_valid(); vv_it++)
                    {
                        VertexHandle v_vh(*vv_it);
                        h_neighbourIdx[offset] = v_vh.idx();
                        offset++;
                    }

                    h_neighbourOffset[count + 1] = offset;

                    count++;
                }
            } else {
                // Potentially very inefficient. I don't care right now
                std::vector<std::set<int>> neighbors(N);
                for (int4 tet : m_sourceTetIndices) {
                    int* t = (int*)&tet;
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 1; j < 4; ++j) {
                            neighbors[t[i]].insert(t[(i + j) % 4]);
                        }
                    }
                }
                uint offset = 0;
                h_neighbourOffset[0] = 0;
                for (int i = 0; i < (int)N; ++i) {
                    int valence = (int)neighbors[i].size();
                    h_numNeighbours[i] = valence;
                    for (auto n : neighbors[i]) {
                        h_neighbourIdx.push_back(n);
                        ++offset;
                    }
                    h_neighbourOffset[i+1] = offset;

                }
            }
            printf("Total Edge count = %d\n", (int)h_neighbourIdx.size());
        }

		void resetGPUMemory()
		{
            std::vector<int> h_numNeighbours, h_neighbourIdx, h_neighbourOffset;
            generateOptEdges(h_numNeighbours, h_neighbourIdx, h_neighbourOffset);

            m_graph = createGraphFromNeighborLists(h_neighbourIdx, h_neighbourOffset);

            uint N = (uint)m_initial.n_vertices();
            std::vector<float3> h_vertexPosFloat3(N);

			for (unsigned int i = 0; i < N; i++)
			{
				const Vec3f& pt = m_initial.point(VertexHandle(i));
				h_vertexPosFloat3[i] = make_float3(pt[0], pt[1], pt[2]);
			}

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

        SimpleMesh* result()
        {
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

        std::unique_ptr<NanoKDTree> generateAccelerationStructure(const SimpleMesh& mesh) {
            unsigned int N = (unsigned int)mesh.n_vertices();

            assert(m_spuriousIndices.size() == m_noisyOffsets.size());
            m_pointCloud.pts.resize(N);
            for (unsigned int i = 0; i < N; i++)
            {   
                auto p = mesh.point(VertexHandle(i));
                m_pointCloud.pts[i] = { p[0], p[1], p[2] };
            }
            std::unique_ptr<NanoKDTree> tree = std::unique_ptr<NanoKDTree>(new NanoKDTree(3 /*dim*/, m_pointCloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */)));
            tree->buildIndex();
            return tree;
        }


        ml::Timer m_timer;
        PointCloud_nanoflann m_pointCloud;
        std::unique_ptr<NanoKDTree> m_targetAccelerationStructure;

        std::mt19937 m_rnd;
        std::vector<int> m_spuriousIndices;;
        std::vector<float3> m_noisyOffsets;

		SimpleMesh m_result;
		SimpleMesh m_initial;
        std::vector<SimpleMesh> m_targets;
        std::vector<float3> m_previousConstraints;
        std::vector<int4> m_sourceTetIndices;

        double m_averageEdgeLength;

        // Current index in solve
        uint m_targetIndex;

		std::shared_ptr<OptImage> m_anglesFloat3;
		std::shared_ptr<OptImage> m_vertexPosTargetFloat3;
        std::shared_ptr<OptImage> m_vertexNormalTargetFloat3;
		std::shared_ptr<OptImage> m_vertexPosFloat3;
		std::shared_ptr<OptImage> m_vertexPosFloat3Urshape;
        std::shared_ptr<OptImage> m_robustWeights;
        std::shared_ptr<OptGraph> m_graph;


        float m_weightFit;
        float m_weightRegMax;
        float m_weightRegMin;
        float m_weightRegFactor;
        float m_weightReg;
        float m_functionTolerance;
        float m_fitSqrt;
        float m_regSqrt;


};
