#include <gtest/gtest.h>
#include <kfusion/warp_field.hpp>
#include <vector>
#include "ceres/ceres.h"
#include "opt/mLibInclude.h"

#include "mLibCore.cpp"
#include "mLibLodePNG.cpp"

#include <gtest/gtest.h>
#include <kfusion/warp_field.hpp>
#include <kfusion/warp_field_optimiser.hpp>
#include <string>
#include "opt/main.h"
#include "opt/CombinedSolver.h"
#include "opt/OpenMesh.h"
#include <kfusion/warp_field.hpp>


TEST(WARP_FIELD_TEST, WarpInit)
{
   kfusion::WarpField warpField;
   std::vector<cv::Vec3f> warp_init;

   warp_init.emplace_back(cv::Vec3f(1,1,1));
   warp_init.emplace_back(cv::Vec3f(1,2,-1));
   warp_init.emplace_back(cv::Vec3f(1,-2,1));
   warp_init.emplace_back(cv::Vec3f(1,-1,-1));
   warp_init.emplace_back(cv::Vec3f(-1,1,5));
   warp_init.emplace_back(cv::Vec3f(-1,1,-1));
   warp_init.emplace_back(cv::Vec3f(-1,-1,1));
   warp_init.emplace_back(cv::Vec3f(-1,-1,-1));
   warp_init.emplace_back(cv::Vec3f(2,-3,-1));

   warpField.init(warp_init);

   for(size_t i = 0; i < warpField.getNodes()->size(); i++)
   {
       auto transform = warpField.getNodes()->at(i).transform;
       auto vertex = warpField.getNodes()->at(i).vertex;
       cv::Vec3f pos;
       transform.getTranslation(pos);
       ASSERT_FLOAT_EQ(vertex[0], pos[0]);
       ASSERT_FLOAT_EQ(vertex[1], pos[1]);
       ASSERT_FLOAT_EQ(vertex[2], pos[2]);
   }
}


TEST(WARP_FIELD_TEST, EnergyDataTest)
{
    const float max_error = 1e-3;

    kfusion::WarpField warpField;
    std::vector<cv::Vec3f> warp_init;

    warp_init.emplace_back(cv::Vec3f(1,1,1));
    warp_init.emplace_back(cv::Vec3f(1,2,-1));
    warp_init.emplace_back(cv::Vec3f(1,-2,1));
    warp_init.emplace_back(cv::Vec3f(1,-1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,1,5));
    warp_init.emplace_back(cv::Vec3f(-1,1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,-1));
    warp_init.emplace_back(cv::Vec3f(2,-3,-1));

    warpField.init(warp_init);

    std::vector<cv::Vec3f> source_vertices;
    source_vertices.emplace_back(cv::Vec3f(-3,-3,-3));
    source_vertices.emplace_back(cv::Vec3f(-2,-2,-2));
    source_vertices.emplace_back(cv::Vec3f(0,0,0));
    source_vertices.emplace_back(cv::Vec3f(2,2,2));
    source_vertices.emplace_back(cv::Vec3f(3,3,3));
    source_vertices.emplace_back(cv::Vec3f(4,4,4));

    std::vector<cv::Vec3f> canonical_normals;
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));

    std::vector<cv::Vec3f> target_vertices;
    target_vertices.emplace_back(cv::Vec3f(-2.95f,-2.95f,-2.95f));
    target_vertices.emplace_back(cv::Vec3f(-1.95f,-1.95f,-1.95f));
    target_vertices.emplace_back(cv::Vec3f(0.05,0.05,0.05));
    target_vertices.emplace_back(cv::Vec3f(2.05,4.05,2.05));
    target_vertices.emplace_back(cv::Vec3f(3.05,3.05,3.05));
    target_vertices.emplace_back(cv::Vec3f(4.5,4.05,6));

    std::vector<cv::Vec3f> target_normals;
    target_normals.emplace_back(cv::Vec3f(0,0,1));
    target_normals.emplace_back(cv::Vec3f(0,0,1));
    target_normals.emplace_back(cv::Vec3f(0,0,1));
    target_normals.emplace_back(cv::Vec3f(0,0,1));
    target_normals.emplace_back(cv::Vec3f(0,0,1));
    target_normals.emplace_back(cv::Vec3f(0,0,1));


    std::vector<cv::Vec3f> initial_source_vertices(source_vertices);
    std::vector<cv::Vec3f> initial_source_normals(canonical_normals);
    CombinedSolverParameters params;
    params.numIter = 20;
    params.nonLinearIter = 15;
    params.linearIter = 250;
    params.useOpt = false;
    params.useOptLM = true;

    kfusion::WarpFieldOptimiser optimiser(&warpField, params);

    optimiser.optimiseWarpData(source_vertices, canonical_normals, target_vertices, target_normals);
    warpField.warp(source_vertices, canonical_normals);

    for(size_t i = 0; i < source_vertices.size(); i++)
    {
        ASSERT_NEAR(source_vertices[i][0], target_vertices[i][0], max_error);
        ASSERT_NEAR(source_vertices[i][1], target_vertices[i][1], max_error);
        ASSERT_NEAR(source_vertices[i][2], target_vertices[i][2], max_error);
        std::cout<<source_vertices[i]<<" "<<target_vertices[i]<<std::endl;
    }

    optimiser.optimiseWarpData(target_vertices, target_normals, initial_source_vertices, initial_source_normals);
    warpField.warp(target_vertices, target_normals);

    for(size_t i = 0; i < source_vertices.size(); i++)
    {
        ASSERT_NEAR(initial_source_vertices[i][0], target_vertices[i][0], max_error);
        ASSERT_NEAR(initial_source_vertices[i][1], target_vertices[i][1], max_error);
        ASSERT_NEAR(initial_source_vertices[i][2], target_vertices[i][2], max_error);
    }
}

