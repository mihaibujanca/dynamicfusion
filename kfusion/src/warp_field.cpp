#include <dual_quaternion.hpp>
#include <knn_point_cloud.hpp>
#include <kfusion/types.hpp>
#include <nanoflann.hpp>
#include "kfusion/warp_field.hpp"
#define VOXEL_SIZE 100

using namespace kfusion;


typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, utils::PointCloud<float>>,
utils::PointCloud<float>,
3 /* dim */
> kd_tree_t;

WarpField::WarpField()
{

}

std::vector<node> WarpField::warp(std::vector<Vec3f> &frame) const
{
    std::vector<utils::DualQuaternion<float>> out_nodes(frame.size());
    for (auto vertex : frame)
    {
        utils::DualQuaternion<float> node = warp(vertex);
        out_nodes.push_back(node);
    }
    return nodes;
}

utils::DualQuaternion<float> kfusion::WarpField::warp(Vec3f point) const
{
    utils::DualQuaternion<float> out;
    utils::PointCloud<float> cloud;
    cloud.pts.resize(nodes.size());
    for(size_t i = 0; i < nodes.size(); i++)
    {
        utils::PointCloud<float>::Point point(nodes[i].vertex.getTranslation().x_,
                                              nodes[i].vertex.getTranslation().y_,
                                              nodes[i].vertex.getTranslation().z_);
        cloud.pts[i] = point;
    }

    kd_tree_t index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    const size_t k = 8; //FIXME: number of neighbours should be a hyperparameter
    std::vector<utils::DualQuaternion<float>> neighbours(k);
    std::vector<size_t> ret_index(k);
    std::vector<float> out_dist_sqr(k);
    nanoflann::KNNResultSet<float> resultSet(k);
    resultSet.init(&ret_index[0], &out_dist_sqr[0]);

    index.findNeighbors(resultSet, point.val, nanoflann::SearchParams(10));

    for (size_t i = 0; i < k; i++)
        neighbours.push_back(nodes[ret_index[i]].vertex);

    utils::DualQuaternion<float> node = DQB(point, VOXEL_SIZE);
    neighbours.clear();
    return node;
}

utils::DualQuaternion<float> WarpField::DQB(Vec3f vertex, float voxel_size) const
{
    utils::DualQuaternion<float> quaternion_sum;
    for(auto node : nodes)
    {
        utils::Quaternion<float> translation = node.vertex.getTranslation();
        Vec3f voxel_center(translation.x_,translation.y_,translation.z_);
        quaternion_sum = quaternion_sum + weighting(vertex, voxel_center, voxel_size) * node.vertex;
    }
    auto norm = quaternion_sum.magnitude();

    return utils::DualQuaternion<float>(quaternion_sum.getRotation() / norm.first,
                                        quaternion_sum.getTranslation() / norm.second);
}

//TODO: KNN already gives the squared distance as well, can pass here instead
float WarpField::weighting(Vec3f vertex, Vec3f voxel_center, float weight) const
{
    float diff = (float) cv::norm(voxel_center, vertex, cv::NORM_L2); // Should this be double?
    return exp(-(diff * diff) / (2 * weight * weight));
}