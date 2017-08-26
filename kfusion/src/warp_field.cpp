#include <dual_quaternion.hpp>
#include <knn_point_cloud.hpp>
#include <kfusion/types.hpp>
#include <nanoflann.hpp>
#include "kfusion/warp_field.hpp"
#include "internal.hpp"
#include "precomp.hpp"
#include <opencv2/core/affine.hpp>
#include <kfusion/optimisation.hpp>

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
    auto voxel_size = kfusion::KinFuParams::default_params_dynamicfusion().volume_size[0] /
                      kfusion::KinFuParams::default_params_dynamicfusion().volume_dims[0];
    for(int i = 0; i < first_frame.rows; i++)
        for(int j = 0; j < first_frame.cols; j++)
        {
            auto point = first_frame.at<Point>(i,j);
            auto norm = normals.at<Normal>(i,j);
            if(!std::isnan(point.x))
            {
                (*nodes)[i*first_frame.cols+j].transform = utils::DualQuaternion<float>(utils::Quaternion<float>(0,point.x, point.y, point.z),
                                                                                     utils::Quaternion<float>(Vec3f(norm.x,norm.y,norm.z)));

                (*nodes)[i*first_frame.cols+j].vertex = Vec3f(point.x,point.y,point.z);
                nodes->at(i*first_frame.cols+j).weight = voxel_size;
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
float WarpField::energy_data(const std::vector<Vec3f> &canonical_vertices,
                             const std::vector<Vec3f> &canonical_normals,
                             const std::vector<Vec3f> &live_vertices,
                             const std::vector<Vec3f> &live_normals,
                             const Intr& intr
)
{
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    int i = 0;
    double *epsilon = new double[getNodes()->size() * 6];
    for(auto v : canonical_vertices)
    {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        if(std::isnan(v[0]))
            continue;
//        else
//            std::cout<<"NOTNAN"<<std::endl;
        cv::Vec3f vl(v[0] + 1,v[0] + 2,v[0] + 3);

        ceres::CostFunction* cost_function =
                DynamicFusionDataEnergy::Create(vl, v, this);
        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 epsilon);
        i++;
    }
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    return 0;
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
        weights[i] = weighting(out_dist_sqr[i], nodes->at(ret_index[i]).weight);
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