#include <gtest/gtest.h>
#include <kfusion/warp_field.hpp>
#include <vector>
#include "ceres/ceres.h"

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
