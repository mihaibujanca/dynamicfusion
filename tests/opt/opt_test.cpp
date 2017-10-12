#include <gtest/gtest.h>
#include <kfusion/warp_field.hpp>
#include <string>
#include <vector>
#include "include/main.h"
#include "include/CombinedSolver.h"
#include "include/OpenMesh.h"
#include "include/LandMarkSet.h"
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LongestEdgeT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LoopT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/CatmullClarkT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/Sqrt3T.hh>

TEST(OPT_WARP_FIELD, EnergyDataTest)
{


    std::string filename = "/home/mihai/Projects/Opt/examples/data/small_armadillo.ply";

    // For now, any model must be accompanied with a identically
    // named (besides the extension, which must be 3 characters) mrk file
    std::string markerFilename = filename.substr(0, filename.size() - 3) + "mrk";
    bool performanceRun = false;

    int subdivisionFactor = 0;
    bool lmOnlyFullSolve = false;

    // Load Constraints
    LandMarkSet markersMesh;
    markersMesh.loadFromFile(markerFilename.c_str());

    std::vector<int>				constraintsIdx;
    std::vector<std::vector<float>> constraintsTarget;

    for (unsigned int i = 0; i < markersMesh.size(); i++)
    {
        constraintsIdx.push_back(markersMesh[i].getVertexIndex());
        constraintsTarget.push_back(markersMesh[i].getPosition());
    }

    SimpleMesh* mesh = new SimpleMesh();
    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename << std::endl;
        exit(1);
    }

    OpenMesh::Subdivider::Uniform::Sqrt3T<SimpleMesh> subdivider;
    // Initialize subdivider
    if (lmOnlyFullSolve) {
        if (subdivisionFactor > 0) {
            subdivider.attach(*mesh);
            subdivider(subdivisionFactor);
            subdivider.detach();
        }
    } else {
        //OpenMesh::Subdivider::Uniform::CatmullClarkT<SimpleMesh> catmull;
        // Execute 1 subdivision steps
        subdivider.attach(*mesh);
        subdivider(1);
        subdivider.detach();
    }
    printf("Faces: %d\nVertices: %d\n", (int)mesh->n_faces(), (int)mesh->n_vertices());

    CombinedSolverParameters params;

    params.numIter = 10;
    params.nonLinearIter = 20;
    params.linearIter = 100;
    params.useOpt = true;
    if (performanceRun) {
        params.useCUDA = false;
        params.useOpt = true;
        params.useOptLM = true;
        params.useCeres = true;
        params.earlyOut = true;
        params.nonLinearIter = 20;
        params.linearIter = 1000;
    }
    if (lmOnlyFullSolve) {
        params.useCUDA = false;
        params.useOpt = false;
        params.useOptLM = true;
        params.earlyOut = true;
        params.linearIter = 1000;// m_image.getWidth()*m_image.getHeight();
        if (mesh->n_vertices() > 100000) {
            params.nonLinearIter = (unsigned int)mesh->n_vertices() / 5000;
        }
    }

    params.optDoublePrecision = false;

    float weightFit = 4.0f;
    float weightReg = 1.0f;
    CombinedSolver solver(mesh, constraintsIdx, constraintsTarget, params, weightFit, weightReg); // input pointcloud
    solver.solveAll();
    SimpleMesh* res = solver.result();
    if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << "out.off" << std::endl;
        exit(1);
    }
}
