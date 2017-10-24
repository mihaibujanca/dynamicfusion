#include <gtest/gtest.h>
#include <kfusion/warp_field.hpp>
#include <vector>
#include "ceres/ceres.h"

TEST(CERES_WARP_FIELD, EnergyDataTest)
{
    const float max_error = 1e-4;

    kfusion::WarpField warpField;
    std::vector<cv::Vec3f> warp_init;
    std::vector<cv::Vec3f> warp_normals;

    for(int i=0; i < KNN_NEIGHBOURS+1; i++)
        warp_normals.emplace_back(cv::Vec3f(0,0,1));

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

    std::vector<cv::Vec3f> canonical_vertices;
    canonical_vertices.emplace_back(cv::Vec3f(-3,-3,-3));
    canonical_vertices.emplace_back(cv::Vec3f(-2,-2,-2));
    canonical_vertices.emplace_back(cv::Vec3f(0,0,0));
    canonical_vertices.emplace_back(cv::Vec3f(2,2,2));
    canonical_vertices.emplace_back(cv::Vec3f(3,3,3));
//    canonical_vertices.emplace_back(cv::Vec3f(4,4,4));

    std::vector<cv::Vec3f> canonical_normals;
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
    canonical_normals.emplace_back(cv::Vec3f(0,0,1));
//    canonical_normals.emplace_back(cv::Vec3f(0,0,1));

    std::vector<cv::Vec3f> live_vertices;
    live_vertices.emplace_back(cv::Vec3f(-2.95f,-2.95f,-2.95f));
    live_vertices.emplace_back(cv::Vec3f(-1.95f,-1.95f,-1.95f));
    live_vertices.emplace_back(cv::Vec3f(0.05,0.05,0.05));
    live_vertices.emplace_back(cv::Vec3f(2.05,2.05,2.05));
    live_vertices.emplace_back(cv::Vec3f(3.05,3.05,3.05));
//    live_vertices.emplace_back(cv::Vec3f(4.5,4.05,6));

    std::vector<cv::Vec3f> live_normals;
    live_normals.emplace_back(cv::Vec3f(0,0,1));
    live_normals.emplace_back(cv::Vec3f(0,0,1));
    live_normals.emplace_back(cv::Vec3f(0,0,1));
    live_normals.emplace_back(cv::Vec3f(0,0,1));
    live_normals.emplace_back(cv::Vec3f(0,0,1));
//    live_normals.emplace_back(cv::Vec3f(0,0,1));

    warpField.energy_data(canonical_vertices, canonical_normals,live_vertices, live_normals);
    warpField.warp(canonical_vertices, canonical_normals);

    for(size_t i = 0; i < canonical_vertices.size(); i++)
    {
        ASSERT_NEAR(canonical_vertices[i][0], live_vertices[i][0], max_error);
        ASSERT_NEAR(canonical_vertices[i][1], live_vertices[i][1], max_error);
        ASSERT_NEAR(canonical_vertices[i][2], live_vertices[i][2], max_error);
    }
}
