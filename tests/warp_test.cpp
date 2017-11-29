#include <gtest/gtest.h>
#include <kfusion/warp_field_optimiser.hpp>
#include <kfusion/warp_field.hpp>
#include <vector>
#include <gtest/gtest.h>
#include <string>

#include "opt/main.h"
#include "opt/CombinedSolver.h"
#include "opt/OpenMesh.h"
#include "opt/mLibInclude.h"
#include "mLibCore.cpp"
#include "mLibLodePNG.cpp"

TEST(WARP_FIELD_TEST, EnergyDataSingleVertexTest)
{
    const float max_error = 1e-3;

    kfusion::WarpField warp_field;
    std::vector<cv::Vec3f> warp_init;

    warp_init.emplace_back(cv::Vec3f(1,1,1));
    warp_init.emplace_back(cv::Vec3f(1,1,-1));
    warp_init.emplace_back(cv::Vec3f(1,-1,1));
    warp_init.emplace_back(cv::Vec3f(1,-1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,1,1));
    warp_init.emplace_back(cv::Vec3f(-1,1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,-1));
    warp_field.init(warp_init);

    std::vector<cv::Vec3f> source_vertices;
    source_vertices.emplace_back(cv::Vec3f(0,0,0));

    std::vector<cv::Vec3f> canonical_normals;
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));

    std::vector<cv::Vec3f> target_vertices;
    target_vertices.emplace_back(cv::Vec3f(0.05,0.05,0.05));

    std::vector<cv::Vec3f> target_normals;
    target_normals.emplace_back(cv::Vec3f(0,0,1));

    CombinedSolverParameters params;
    params.numIter = 20;
    params.nonLinearIter = 15;
    params.linearIter = 250;
    params.useOpt = false;
    params.useOptLM = true;
    params.earlyOut = true;

    kfusion::WarpFieldOptimiser optimiser(&warp_field, params);

    optimiser.optimiseWarpData(source_vertices, canonical_normals, target_vertices, target_normals);
    warp_field.warp(source_vertices, canonical_normals);

    for(int i = 0; i < warp_field.getNodes()->size(); i++)
    {
        auto t = warp_field.getNodes()->at(i).transform.getTranslation();
        std::cout<< t <<std::endl;
    }

    for(size_t i = 0; i < source_vertices.size(); i++)
    {
        ASSERT_NEAR(source_vertices[i][0], target_vertices[i][0], max_error);
        ASSERT_NEAR(source_vertices[i][1], target_vertices[i][1], max_error);
        ASSERT_NEAR(source_vertices[i][2], target_vertices[i][2], max_error);
    }

}


TEST(WARP_FIELD_TEST, EnergyDataRigidTest)
{
    const float max_error = 1e-3;

    kfusion::WarpField warp_field;
    std::vector<cv::Vec3f> warp_init;

    warp_init.emplace_back(cv::Vec3f(1,1,1));
    warp_init.emplace_back(cv::Vec3f(1,1,-1));
    warp_init.emplace_back(cv::Vec3f(1,-1,1));
    warp_init.emplace_back(cv::Vec3f(1,-1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,1,1));
    warp_init.emplace_back(cv::Vec3f(-1,1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,-1));
    warp_field.init(warp_init);

    std::vector<cv::Vec3f> source_vertices;
    source_vertices.emplace_back(cv::Vec3f(-3,-3,-3));
    source_vertices.emplace_back(cv::Vec3f(-2,-2,-2));
    source_vertices.emplace_back(cv::Vec3f(0,0,0));
    source_vertices.emplace_back(cv::Vec3f(2,2,2));
    source_vertices.emplace_back(cv::Vec3f(3,3,3));

    std::vector<cv::Vec3f> canonical_normals;
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));

    std::vector<cv::Vec3f> target_vertices;
    target_vertices.emplace_back(cv::Vec3f(-2.95f,-2.95f,-2.95f));
    target_vertices.emplace_back(cv::Vec3f(-1.95f,-1.95f,-1.95f));
    target_vertices.emplace_back(cv::Vec3f(0.05,0.05,0.05));
    target_vertices.emplace_back(cv::Vec3f(2.05,2.05,2.05));
    target_vertices.emplace_back(cv::Vec3f(3.05,3.05,3.05));

    std::vector<cv::Vec3f> target_normals;
    target_normals.emplace_back(cv::Vec3f(0,0,1));
    target_normals.emplace_back(cv::Vec3f(0,0,1));
    target_normals.emplace_back(cv::Vec3f(0,0,1));
    target_normals.emplace_back(cv::Vec3f(0,0,1));
    target_normals.emplace_back(cv::Vec3f(0,0,1));

    CombinedSolverParameters params;
    params.numIter = 20;
    params.nonLinearIter = 15;
    params.linearIter = 250;
    params.useOpt = false;
    params.useOptLM = true;
    params.earlyOut = false;

    kfusion::WarpFieldOptimiser optimiser(&warp_field, params);

    optimiser.optimiseWarpData(source_vertices, canonical_normals, target_vertices, target_normals);
    warp_field.warp(source_vertices, canonical_normals);


    for(int i = 0; i < warp_field.getNodes()->size(); i++)
    {
        auto t = warp_field.getNodes()->at(i).transform.getTranslation();
        std::cout<< t <<std::endl;
    }

    for(size_t i = 0; i < source_vertices.size(); i++)
    {
        ASSERT_NEAR(source_vertices[i][0], target_vertices[i][0], max_error);
        ASSERT_NEAR(source_vertices[i][1], target_vertices[i][1], max_error);
        ASSERT_NEAR(source_vertices[i][2], target_vertices[i][2], max_error);
    }
}


TEST(WARP_FIELD_TEST, WarpAndReverseTest)
{
    const float max_error = 1e-3;

    kfusion::WarpField warp_field;
    std::vector<cv::Vec3f> warp_init;

    warp_init.emplace_back(cv::Vec3f(1,1,1));
    warp_init.emplace_back(cv::Vec3f(1,1,-1));
    warp_init.emplace_back(cv::Vec3f(1,-1,1));
    warp_init.emplace_back(cv::Vec3f(1,-1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,1,1));
    warp_init.emplace_back(cv::Vec3f(-1,1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,-1));
    warp_field.init(warp_init);

    std::vector<cv::Vec3f> source_vertices;
    source_vertices.emplace_back(cv::Vec3f(-3,-3,-3));
    source_vertices.emplace_back(cv::Vec3f(-2,-2,-2));
    source_vertices.emplace_back(cv::Vec3f(0,0,0));
    source_vertices.emplace_back(cv::Vec3f(2,2,2));
    source_vertices.emplace_back(cv::Vec3f(3,3,3));

    std::vector<cv::Vec3f> canonical_normals;
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));

    std::vector<cv::Vec3f> target_vertices;
    target_vertices.emplace_back(cv::Vec3f(-2.95f,-2.95f,-2.95f));
    target_vertices.emplace_back(cv::Vec3f(-1.95f,-1.95f,-1.95f));
    target_vertices.emplace_back(cv::Vec3f(0.05,0.05,0.05));
    target_vertices.emplace_back(cv::Vec3f(2.05,2.05,2.05));
    target_vertices.emplace_back(cv::Vec3f(3.05,3.05,3.05));

    std::vector<cv::Vec3f> target_normals;
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

    kfusion::WarpFieldOptimiser optimiser(&warp_field, params);

    optimiser.optimiseWarpData(source_vertices, canonical_normals, target_vertices, target_normals);
    warp_field.warp(source_vertices, canonical_normals);

//    for(size_t i = 0; i < source_vertices.size(); i++)
//    {
//        ASSERT_NEAR(source_vertices[i][0], target_vertices[i][0], max_error);
//        ASSERT_NEAR(source_vertices[i][1], target_vertices[i][1], max_error);
//        ASSERT_NEAR(source_vertices[i][2], target_vertices[i][2], max_error);
//    }

    auto sum = kfusion::utils::Quaternion<float>();
    for(int i = 0; i < warp_field.getNodes()->size(); i++)
    {
        auto t = warp_field.getNodes()->at(i).transform.getTranslation();
        sum = sum + t;
        std::cout<< t <<std::endl;
    }
    optimiser.optimiseWarpData(target_vertices, target_normals, initial_source_vertices, initial_source_normals);
    warp_field.warp(target_vertices, target_normals);

    for(size_t i = 0; i < source_vertices.size(); i++)
    {
        ASSERT_NEAR(initial_source_vertices[i][0], target_vertices[i][0], max_error);
        ASSERT_NEAR(initial_source_vertices[i][1], target_vertices[i][1], max_error);
        ASSERT_NEAR(initial_source_vertices[i][2], target_vertices[i][2], max_error);
    }

    for(int i = 0; i < warp_field.getNodes()->size(); i++)
    {
        auto t = warp_field.getNodes()->at(i).transform.getTranslation();
        sum = sum + t;
        std::cout<< t <<std::endl;
    }
    ASSERT_NEAR(sum.x_, 0, max_error);
    ASSERT_NEAR(sum.y_, 0, max_error);
    ASSERT_NEAR(sum.z_, 0, max_error);
    std::cout<<std::endl<<std::endl<<"Final sum:"<<sum<<std::endl;
}

TEST(WARP_FIELD_TEST, MultipleNodesTest)
{
    const float max_error = 1e-3;

    kfusion::WarpField warp_field;
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
    warp_init.emplace_back(cv::Vec3f(-3,-3,-2));
    warp_init.emplace_back(cv::Vec3f(2,-3,3));
    warp_init.emplace_back(cv::Vec3f(2,2,4));
    warp_field.init(warp_init);

    std::vector<cv::Vec3f> source_vertices;
    source_vertices.emplace_back(cv::Vec3f(-3,-3,-3));
    source_vertices.emplace_back(cv::Vec3f(-2,-2,-2));
    source_vertices.emplace_back(cv::Vec3f(0,0,0));
    source_vertices.emplace_back(cv::Vec3f(2,2,2));
    source_vertices.emplace_back(cv::Vec3f(3,3,3));
    source_vertices.emplace_back(cv::Vec3f(3,3,3));

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
    target_vertices.emplace_back(cv::Vec3f(0.1,0.1,0.1));
    target_vertices.emplace_back(cv::Vec3f(2,2,2));
    target_vertices.emplace_back(cv::Vec3f(3.05,3.05,3.05));
    target_vertices.emplace_back(cv::Vec3f(3.05,3.05,3.05));

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

    kfusion::WarpFieldOptimiser optimiser(&warp_field, params);
    optimiser.optimiseWarpData(source_vertices, canonical_normals, target_vertices, target_normals);
    warp_field.warp(source_vertices, canonical_normals);

    for(size_t i = 0; i < source_vertices.size(); i++)
    {
        ASSERT_NEAR(source_vertices[i][0], target_vertices[i][0], max_error);
        ASSERT_NEAR(source_vertices[i][1], target_vertices[i][1], max_error);
        ASSERT_NEAR(source_vertices[i][2], target_vertices[i][2], max_error);
    }
}



TEST(WARP_FIELD_TEST, NonRigidTest)
{
    const float max_error = 1e-3;

    kfusion::WarpField warp_field;
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
    warp_field.init(warp_init);

    std::vector<cv::Vec3f> source_vertices;
    source_vertices.emplace_back(cv::Vec3f(-3,-3,-3));
    source_vertices.emplace_back(cv::Vec3f(-2,-2,-2));
    source_vertices.emplace_back(cv::Vec3f(0,0,0));
    source_vertices.emplace_back(cv::Vec3f(2,2,2));
    source_vertices.emplace_back(cv::Vec3f(3,3,3));
    source_vertices.emplace_back(cv::Vec3f(3,3,3));

    std::vector<cv::Vec3f> canonical_normals;
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));

    std::vector<cv::Vec3f> target_vertices;
    target_vertices.emplace_back(cv::Vec3f(-2.95f,-3.f,-2.95f));
    target_vertices.emplace_back(cv::Vec3f(-1.95f,-1.95f,-2.f));
    target_vertices.emplace_back(cv::Vec3f(0.1,0.1,0.1));
    target_vertices.emplace_back(cv::Vec3f(2,2.5f,2));
    target_vertices.emplace_back(cv::Vec3f(3.05,3.05,3.05));
    target_vertices.emplace_back(cv::Vec3f(3.05,3.05,3.05));

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

    kfusion::WarpFieldOptimiser optimiser(&warp_field, params);
    optimiser.optimiseWarpData(source_vertices, canonical_normals, target_vertices, target_normals);
    warp_field.warp(source_vertices, canonical_normals);

    for(size_t i = 0; i < source_vertices.size(); i++)
    {
        ASSERT_NEAR(source_vertices[i][0], target_vertices[i][0], max_error);
        ASSERT_NEAR(source_vertices[i][1], target_vertices[i][1], max_error);
        ASSERT_NEAR(source_vertices[i][2], target_vertices[i][2], max_error);
    }
}
