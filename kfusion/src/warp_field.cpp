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
    nodes = new std::vector<deformation_node>();
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
    nodes->resize(first_frame.cols * first_frame.rows);

    for(int i = 0; i < first_frame.rows; i++)
        for(int j = 0; j < first_frame.cols; j++)
        {
            auto point = first_frame.at<Point>(i,j);
            auto norm = normals.at<Normal>(i,j);
            if(!std::isnan(point.x))
            {
                nodes->at(i*first_frame.cols+j).transform = utils::DualQuaternion<float>(utils::Quaternion<float>(0,point.x, point.y, point.z),
                                                                                     utils::Quaternion<float>(Vec3f(norm.x,norm.y,norm.z)));

                nodes->at(i*first_frame.cols+j).vertex = Vec3f(point.x,point.y,point.z);
                nodes->at(i*first_frame.cols+j).weight = VOXEL_SIZE;
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
}

/**
 * \brief
 * \param frame
 * \param pose
 * \param tsdfVolume
 */
float WarpField::energy_data(const std::vector<Vec3f> &warped_vertices,
                             const std::vector<Vec3f> &warped_normals,
                             const Intr& intr
)
{
    float total_energy = 0;
//get dual quaternion for that vertex/normal and K neighbouring dual quaternions

    int i = 0;
    for(auto v : warped_vertices)
    {
        if(std::isnan(warped_normals[i][0]) || std::isnan(v[0]))
            continue;
        Vec3f vl(v[0] * intr.fx / -v[2] + intr.cx, v[1] * intr.fy / v[2] + intr.cy, v[2]);
        const float energy = tukeyPenalty(warped_normals[i].dot(v - vl)); // normal (warp - live)
        total_energy += energy;
        i++;
    }
    return total_energy;
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
 *
 * In the paper, a value of 0.01 is suggested for c
 */
float WarpField::tukeyPenalty(float x, float c) const
{
    return std::abs(x) <= c ? x * std::pow((1 - (x * x) / (c * c)), 2) : 0.0f;
}

/**
 * Huber penalty function, implemented as described in https://en.wikipedia.org/wiki/Huber_loss
 * In the paper, a value of 0.0001 is suggested for delta
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
void WarpField::warp(std::vector<Vec3f>& points, std::vector<Vec3f>& normals) const
{
    int i = 0;
    for (auto& point : points)
    {
        if(std::isnan(point[0]) || std::isnan(normals[i][0]))
            continue;
        KNN(point);
        utils::DualQuaternion<float> dqb = DQB(point);
        dqb.transform(point);
        point = warp_to_live * point;

        dqb.transform(normals[i]);
        normals[i] = warp_to_live * normals[i];
        i++;
    }
}


/**
 * Modifies the
 * @param points
 */
void WarpField::warp(cuda::Cloud& points) const
{
    int i = 0;
    int nans = 0;
//    for (auto& point : points)
//    {
//        i++;
//        if(std::isnan(point[0]) || std::isnan(point[1]) || std::isnan(point[2]))
//        {
//            nans++;
//            continue;
//        }
//        KNN(point);
//        utils::DualQuaternion<float> dqb = DQB(point);
//        point = warp_to_live * point; // Apply T_lw first. Is this not inverse of the pose?
//        dqb.transform(point);
//    }
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
        quaternion_sum = quaternion_sum + weighting(out_dist_sqr[ret_index[i]], nodes->at(ret_index[i]).weight) *
                                                  nodes->at(ret_index[i]).transform;

    auto norm = quaternion_sum.magnitude();

    return utils::DualQuaternion<float>(quaternion_sum.getRotation() / norm.first,
                                        quaternion_sum.getTranslation() / norm.second);
}


/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
utils::DualQuaternion<float> WarpField::DQB(const Vec3f& vertex, double epsilon[KNN_NEIGHBOURS * 6]) const
{
    if(epsilon == NULL)
    {
        std::cerr<<"Invalid pointer in DQB"<<std::endl;
        exit(-1);
    }
    utils::DualQuaternion<float> quaternion_sum;
    utils::DualQuaternion<float> eps;
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
    {
        // epsilon [0:2] is rotation [3:5] is translation
        eps.from_twist(epsilon[i*6],epsilon[i*6 + 1],epsilon[i*6 + 2],epsilon[i*6 + 3],epsilon[i*6 + 4],epsilon[i*6 + 5]);
        quaternion_sum = quaternion_sum + weighting(out_dist_sqr[ret_index[i]], nodes->at(ret_index[i]).weight) *
                                                  nodes->at(ret_index[i]).transform * eps;
    }

    auto norm = quaternion_sum.magnitude();

    return utils::DualQuaternion<float>(quaternion_sum.getRotation() / norm.first,
                                        quaternion_sum.getTranslation() / norm.second);
}

/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
void WarpField::getWeightsAndUpdateKNN(const Vec3f& vertex, float weights[KNN_NEIGHBOURS])
{
    KNN(vertex);
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
        // epsilon [0:2] is rotation [3:5] is translation
        weights[i] = weighting(out_dist_sqr[ret_index[i]], nodes->at(ret_index[i]).weight);
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
    return nodes;
}

/**
 * \brief
 * \return
 */
void WarpField::buildKDTree()
{
    //    Build kd-tree with current warp nodes.
    cloud.pts.resize(nodes->size());
    for(size_t i = 0; i < nodes->size(); i++)
        nodes->at(i).transform.getTranslation(cloud.pts[i]);
    index->buildIndex();
}

const cv::Mat WarpField::getNodesAsMat() const
{
    cv::Mat matrix(1, nodes->size(), CV_32FC3);
    for(int i = 0; i < nodes->size(); i++)
        matrix.at<cv::Vec3f>(i) = nodes->at(i).vertex;
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