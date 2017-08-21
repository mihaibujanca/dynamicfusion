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
std::vector<utils::DualQuaternion<float>> neighbours; //THIS SHOULD BE SOMEWHERE ELSE BUT TOO SLOW TO REINITIALISE
utils::PointCloud cloud;

WarpField::WarpField()
{
    index = new kd_tree_t(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    ret_index = std::vector<size_t>(KNN_NEIGHBOURS);
    out_dist_sqr = std::vector<float>(KNN_NEIGHBOURS);
    resultSet = new nanoflann::KNNResultSet<float>(KNN_NEIGHBOURS);
    resultSet->init(&ret_index[0], &out_dist_sqr[0]);
    neighbours = std::vector<utils::DualQuaternion<float>>(KNN_NEIGHBOURS);

}

WarpField::~WarpField()
{}

/**
 *
 * @param frame
 * \note The pose is assumed to be the identity, as this is the first frame
 */
// maybe remove this later and do everything as part of energy since all this code is written twice. Leave it for now.
void WarpField::init(const cv::Mat& first_frame, const cv::Mat& normals)
{
    assert(first_frame.rows == normals.rows);
    assert(first_frame.cols == normals.cols);
    nodes.resize(first_frame.cols * first_frame.rows);

    for(int i = 0; i < first_frame.rows; i++)
        for(int j = 0; j < first_frame.cols; j++)
        {
            auto point = first_frame.at<Point>(i,j);
            auto norm = normals.at<Normal>(i,j);
            if(!std::isnan(point.x))
            {
                nodes[i*first_frame.cols+j].transform = utils::DualQuaternion<float>(utils::Quaternion<float>(0,point.x, point.y, point.z),
                                                                                     utils::Quaternion<float>(Vec3f(norm.x,norm.y,norm.z)));

                nodes[i*first_frame.cols+j].vertex = Vec3f(point.x,point.y,point.z);
                nodes[i*first_frame.cols+j].weight = VOXEL_SIZE;
            }
            else
            {
                nodes[i*first_frame.cols+j].valid = false;
            }
        }
    buildKDTree();
}


/**
 * \brief
 * \param frame
 * \param normals
 * \param pose
 * \param tsdfVolume
 * \param edges
 */
void WarpField::energy(const cuda::Cloud &frame,
                       const cuda::Normals &normals,
                       const Affine3f &pose,
                       const cuda::TsdfVolume &tsdfVolume,
                       const std::vector<std::pair<utils::DualQuaternion<float>, utils::DualQuaternion<float>>> &edges
)
{
    assert(normals.cols()==frame.cols());
    assert(normals.rows()==frame.rows());

    int cols = frame.cols();

    std::vector<Point, std::allocator<Point>> cloud_host(size_t(frame.rows()*frame.cols()));
    frame.download(cloud_host, cols);

    std::vector<Normal, std::allocator<Normal>> normals_host(size_t(normals.rows()*normals.cols()));
    normals.download(normals_host, cols);
    for(size_t i = 0; i < cloud_host.size() && i < nodes.size(); i++)
    {
        auto point = cloud_host[i];
        auto norm = normals_host[i];
        if(!std::isnan(point.x))
        {

        }
        else
        {
            std::cout<<"NANS"<<std::endl;
            break;
        }
    }
}

/**
 * \brief
 * \param frame
 * \param pose
 * \param tsdfVolume
 */
void WarpField::energy_data(const cuda::Depth &frame,
                            const Affine3f &pose,
                            const cuda::TsdfVolume &tsdfVolume
)
{


}

/**
 * \brief
 * \param edges
 */
void WarpField::energy_reg(const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
        kfusion::utils::DualQuaternion<float>>> &edges)
{

}

/**
 * Tukey loss function as described in http://web.as.uky.edu/statistics/users/pbreheny/764-F11/notes/12-1.pdf
 * \param x
 * \param c
 * \return
 *
 * \note
 * The value c = 4.685 is usually used for this loss function, and
 * it provides an asymptotic efficiency 95% that of linear
 * regression for the normal distribution
 */
float WarpField::tukeyPenalty(float x, float c) const
{
    return std::abs(x) <= c ? x * std::pow((1 - (x * x) / (c * c)), 2) : 0.0;
}

/**
 * Huber penalty function, implemented as described in https://en.wikipedia.org/wiki/Huber_loss
 * \param a
 * \param delta
 * \return
 */
float WarpField::huberPenalty(float a, float delta) const
{
    return std::abs(a) <= delta ? a * a / 2 : delta * std::abs(a) - delta * delta / 2;
}

/**
 * Modifies the
 * @param points
 */
void WarpField::warp(std::vector<Vec3f>& points) const
{
    int i = 0;
    int nans = 0;
    for (auto& point : points)
    {
        i++;
        if(std::isnan(point[0]) || std::isnan(point[1]) || std::isnan(point[2]))
        {
            nans++;
            continue;
        }
        KNN(point);
        utils::DualQuaternion<float> dqb = DQB(point);
        point = warp_to_live * point; // Apply T_lw first. Is this not inverse of the pose?
//        dqb.transform(point);
    }
}


/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
utils::DualQuaternion<float> WarpField::DQB(const Vec3f& vertex) const
{
    utils::DualQuaternion<float> quaternion_sum;
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
        //FIXME: accessing nodes[ret_index[i]].transform VERY SLOW. Assignment also very slow
        quaternion_sum = quaternion_sum + weighting(out_dist_sqr[ret_index[i]], nodes[ret_index[i]].weight) * nodes[ret_index[i]].transform;

    auto norm = quaternion_sum.magnitude();

    return utils::DualQuaternion<float>(quaternion_sum.getRotation() / norm.first,
                                        quaternion_sum.getTranslation() / norm.second);
}

/**
 * \brief
 * \param squared_dist
 * \param weight
 * \return
 */
float WarpField::weighting(float squared_dist, float weight) const
{
    return (float) exp(-squared_dist / (2 * weight * weight));
}

/**
 * \brief
 * \return
 */
void WarpField::KNN(Vec3f point) const
{
    index->findNeighbors(*resultSet, point.val, nanoflann::SearchParams(10));
}

/**
 * \brief
 * \return
 */
const std::vector<deformation_node>* WarpField::getNodes() const
{
    return &nodes;
}

/**
 * \brief
 * \return
 */
void WarpField::buildKDTree()
{
    //    Build kd-tree with current warp nodes.
    cloud.pts.resize(nodes.size());
    for(size_t i = 0; i < nodes.size(); i++)
        nodes[i].transform.getTranslation(cloud.pts[i]);
    index->buildIndex();
}

//TODO: This can be optimised
const cv::Mat WarpField::getNodesAsMat() const
{
    cv::Mat matrix(1, nodes.size(), CV_32FC3);
    for(int i = 0; i < nodes.size(); i++)
        matrix.at<cv::Vec3f>(i) = nodes[i].vertex;
    return matrix;
}

/**
 * \brief
 */
void WarpField::clear()
{

}
void WarpField::setWarpToLive(const Affine3f &pose)
{
    warp_to_live = pose;
}