#include <dual_quaternion.hpp>
#include <knn_point_cloud.hpp>
#include <kfusion/types.hpp>
#include <nanoflann.hpp>
#include "kfusion/warp_field.hpp"
#include "internal.hpp"
#include "precomp.hpp"
#include <opencv2/core/affine.hpp>
#define VOXEL_SIZE 100

using namespace kfusion;


WarpField::WarpField()
{}

WarpField::~WarpField()
{}
/**
 *
 * @param frame
 * \note The pose is assumed to be the identity, as this is the first frame
 */
// maybe remove this later and do everything as part of energy since all this code is written twice. Leave it for now.
void WarpField::init(const cuda::Cloud &frame, const cuda::Normals &normals)
{
    assert(normals.cols()==frame.cols());
    assert(normals.rows()==frame.rows());
    int cols = frame.cols();

    std::vector<Point, std::allocator<Point>> cloud_host(size_t(frame.rows()*frame.cols()));
    frame.download(cloud_host, cols);

    std::vector<Normal, std::allocator<Normal>> normals_host(size_t(normals.rows()*normals.cols()));
    normals.download(normals_host, cols);

    nodes.reserve(cloud_host.size());

    for(size_t i = 0; i < cloud_host.size() && i < nodes.size(); i++) // FIXME: for now just stop at the number of nodes
    {
        auto point = cloud_host[i];
        auto norm = normals_host[i];
        if(!std::isnan(point.x))
        {
            // TODO:    transform by pose
            Vec3f position(point.x,point.y,point.z);
            Vec3f normal(norm.x,norm.y,norm.z);

            utils::DualQuaternion<float> dualQuaternion(utils::Quaternion<float>(0,position[0], position[1], position[2]),
                                                        utils::Quaternion<float>(normal));

            nodes[i].vertex = position;
            nodes[i].transform = dualQuaternion;
        }
        else
        {
            //    FIXME: will need to deal with the case when we get NANs
            std::cout<<"NANS"<<std::endl;
            break;
        }
    }
}

void WarpField::energy(const cuda::Cloud &frame,
                       const cuda::Normals &normals,
                       const Affine3f &pose,
                       const cuda::TsdfVolume &tsdfVolume,
                       const std::vector<std::pair<utils::DualQuaternion<float>, utils::DualQuaternion<float>>> &edges
)
{
    assert(normals.cols()==frame.cols());
    assert(normals.rows()==frame.rows());

    //  TODO: proper implementation. At the moment just initialise the positions with the old Quaternion positions
    for(auto node : nodes)
        node.transform.getTranslation(node.vertex);


    int cols = frame.cols();

    std::vector<Point, std::allocator<Point>> cloud_host(size_t(frame.rows()*frame.cols()));
    frame.download(cloud_host, cols);

    std::vector<Normal, std::allocator<Normal>> normals_host(size_t(normals.rows()*normals.cols()));
    normals.download(normals_host, cols);
    for(size_t i = 0; i < cloud_host.size() && i < nodes.size(); i++) // FIXME: for now just stop at the number of nodes
    {
        auto point = cloud_host[i];
        auto norm = normals_host[i];
        if(!std::isnan(point.x))
        {
            // TODO:    transform by pose
            Vec3f position(point.x,point.y,point.z);
            Vec3f normal(norm.x,norm.y,norm.z);

            utils::DualQuaternion<float> dualQuaternion(utils::Quaternion<float>(0,position[0], position[1], position[2]),
                                                        utils::Quaternion<float>(normal));
            nodes[i].transform = dualQuaternion;
        }
        else
        {
            //    FIXME: will need to deal with the case when we get NANs
            std::cout<<"NANS"<<std::endl;
            break;
        }
    }


}
/**
 * Modifies the
 * @param cloud_host
 * @param normals_host
 */
void WarpField::warp(std::vector<Point, std::allocator<Point>>& cloud_host,
                     std::vector<Point, std::allocator<Point>>& normals_host) const
{

    for (auto point : cloud_host)
    {
        Vec3f vertex(point.x,point.y,point.z);
        utils::DualQuaternion<float> node = warp(vertex);
//       Apply the transformation to the vertex and the normal
    }
}

utils::DualQuaternion<float> kfusion::WarpField::warp(Vec3f point) const
{
    utils::DualQuaternion<float> out;
    utils::PointCloud cloud;
    cloud.pts.resize(nodes.size());
    for(size_t i = 0; i < nodes.size(); i++)
        nodes[i].transform.getTranslation(cloud.pts[i]);

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
        neighbours.push_back(nodes[ret_index[i]].transform);

    utils::DualQuaternion<float> node = DQB(point, VOXEL_SIZE);
    neighbours.clear();
    return node;
}

utils::DualQuaternion<float> WarpField::DQB(Vec3f vertex, float voxel_size) const
{
    utils::DualQuaternion<float> quaternion_sum;
    for(auto node : nodes)
    {
        utils::Quaternion<float> translation = node.transform.getTranslation();
        Vec3f voxel_center(translation.x_,translation.y_,translation.z_);
        quaternion_sum = quaternion_sum + weighting(vertex, voxel_center, voxel_size) * node.transform;
    }
    auto norm = quaternion_sum.magnitude();

    return utils::DualQuaternion<float>(quaternion_sum.getRotation() / norm.first,
                                        quaternion_sum.getTranslation() / norm.second);
}

//TODO: KNN already gives the squared distance as well, can pass here instead
float WarpField::weighting(Vec3f vertex, Vec3f voxel_center, float weight) const
{
    double diff = cv::norm(voxel_center, vertex, cv::NORM_L2);
    return (float) exp(-(diff * diff) / (2 * weight * weight)); // FIXME: Not exactly clean
}