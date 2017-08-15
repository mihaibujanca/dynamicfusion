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
void WarpField::init(const cv::Mat& cloud_host, const cv::Mat& normals_host)
{
    assert(cloud_host.rows == normals_host.rows);
    assert(cloud_host.cols == normals_host.cols);
    nodes.resize(cloud_host.cols * cloud_host.rows);

    for(int i = 0; i < cloud_host.rows; i++) // FIXME: for now just stop at the number of nodes
        for(int j = 0; j < cloud_host.cols; j++) // FIXME: for now just stop at the number of nodes
        {
            auto point = cloud_host.at<Point>(i,j);
            auto norm = normals_host.at<Normal>(i,j);
            if(!std::isnan(point.x))
            {
                nodes[i*cloud_host.cols+j].transform = utils::DualQuaternion<float>(utils::Quaternion<float>(0,point.x, point.y, point.z),
                                                                                    utils::Quaternion<float>(Vec3f(norm.x,norm.y,norm.z)));

                nodes[i*cloud_host.cols+j].vertex = Vec3f(point.x,point.y,point.z);
            }
            else
            {
//                nodes[i*cloud_host.cols+j].valid = false;
            }
        }
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
            nodes[i].transform = utils::DualQuaternion<float>(utils::Quaternion<float>(0, point.x, point.y, point.z),
                                                              utils::Quaternion<float>(Vec3f(norm.x,norm.y,norm.z)));
        else
        {
            //    FIXME: will need to deal with the case when we get NANs
            std::cout<<"NANS"<<std::endl;
            break;
        }
    }
}

/**
 * \brief
 * \param frame
 * \param normals
 * \param pose
 * \param tsdfVolume
 * \param edges
 */
void WarpField::energy_temp(const Affine3f &pose)
{
    for(auto &node : nodes)
        node.vertex = pose * node.vertex;
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
 * @param cloud_host
 * @param normals_host
 */
void WarpField::warp(std::vector<Point, std::allocator<Point>>& cloud_host,
                     std::vector<Point, std::allocator<Point>>& normals_host) const
{
    assert(cloud_host.size() == normals_host.size());

    for (auto point : cloud_host)
    {
        Vec3f vertex(point.x,point.y,point.z);
//        utils::DualQuaternion<float> node = warp(vertex);
        //       Apply the transformation to the vertex and the normal
    }
}

/**
 * Modifies the
 * @param cloud_host
 * @param normals_host
 */
void WarpField::warp(std::vector<Vec3f>& cloud_host) const
{
    cloud.pts.resize(nodes.size());
    for(size_t i = 0; i < nodes.size(); i++)
        nodes[i].transform.getTranslation(cloud.pts[i]);

    index->buildIndex();
    for (auto& point : cloud_host)
    {
        KNN(point);
        utils::DualQuaternion<float> dqb;// = DQB(point, VOXEL_SIZE);
        dqb.transform(point);
    }
}

/**
 * \brief
 * \param vertex
 * \param voxel_size
 * \return
 */
utils::DualQuaternion<float> WarpField::DQB(Vec3f vertex, float voxel_size) const
{
    utils::DualQuaternion<float> quaternion_sum;
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
        //FIXME: accessing nodes[ret_index[i]].transform VERY SLOW. Assignment also very slow
        quaternion_sum = quaternion_sum + weighting(out_dist_sqr[ret_index[i]], voxel_size) * nodes[ret_index[i]].transform;

    auto norm = quaternion_sum.magnitude();

    return utils::DualQuaternion<float>(quaternion_sum.getRotation() / norm.first,
                                        quaternion_sum.getTranslation() / norm.second);
}

/**
 * \brief
 * \param vertex
 * \param voxel_center
 * \param weight
 * \return
 */
float WarpField::weighting(Vec3f vertex, Vec3f voxel_center, float weight) const
{
    double diff = cv::norm(voxel_center, vertex, cv::NORM_L2);
    return (float) exp(-(diff * diff) / (2 * weight * weight)); // FIXME: Not exactly clean
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
