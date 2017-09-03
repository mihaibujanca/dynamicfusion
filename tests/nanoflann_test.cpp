#include <nanoflann/nanoflann.hpp>
#include <iostream>
#include "../kfusion/src/utils/knn_point_cloud.hpp"
#define KNN_NEIGHBOURS 8
typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, kfusion::utils::PointCloud>,
        kfusion::utils::PointCloud,
        3 /* dim */
> kd_tree_t;

int main(int argc, char** argv) {

    kd_tree_t* index_;
    nanoflann::KNNResultSet<float> *resultSet_;
    std::vector<float> out_dist_sqr_(KNN_NEIGHBOURS);
    std::vector<size_t> ret_index_(KNN_NEIGHBOURS);
    kfusion::utils::PointCloud cloud;

    index_ = new kd_tree_t(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    resultSet_ = new nanoflann::KNNResultSet<float>(KNN_NEIGHBOURS);
    resultSet_->init(&ret_index_[0], &out_dist_sqr_[0]);

    std::vector<cv::Vec3f> warp_init;
    warp_init.emplace_back(cv::Vec3f(1,1,1));
    warp_init.emplace_back(cv::Vec3f(1,1,-1));
    warp_init.emplace_back(cv::Vec3f(1,-1,1));
    warp_init.emplace_back(cv::Vec3f(1,-1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,1,1));
    warp_init.emplace_back(cv::Vec3f(-1,1,-1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,1));
    warp_init.emplace_back(cv::Vec3f(-1,-1,-1));
    cloud.pts = warp_init;

    std::vector<cv::Vec3f> canonical_vertices;
    canonical_vertices.emplace_back(cv::Vec3f(-1,-1,-1));
    canonical_vertices.emplace_back(cv::Vec3f(0,0,0));
    canonical_vertices.emplace_back(cv::Vec3f(1,1,1));
    canonical_vertices.emplace_back(cv::Vec3f(2,2,2));
    canonical_vertices.emplace_back(cv::Vec3f(3,3,3));

    index_->buildIndex();

    for(auto v : canonical_vertices)
    {
        index_->findNeighbors(*resultSet_, v.val, nanoflann::SearchParams(10));
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
            std::cout<<ret_index_[i]<<" ";
        std::cout<<std::endl;
    }

    return 0;
}
