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
                nodes->at(i*first_frame.cols+j).transform = utils::DualQuaternion<float>(utils::Quaternion<float>(0,point.x, point.y, point.z),
                                                                                         utils::Quaternion<float>(Vec3f(norm.x,norm.y,norm.z)));

                nodes->at(i*first_frame.cols+j).vertex = Vec3f(point.x,point.y,point.z);
                nodes->at(i*first_frame.cols+j).weight = voxel_size;
            }
        }
    buildKDTree();
}

/**
 *
 * @param frame
 * \note The pose is assumed to be the identity, as this is the first frame
 */
// maybe remove this later and do everything as part of energy since all this code is written twice. Leave it for now.
void WarpField::init(const std::vector<Vec3f>& first_frame, const std::vector<Vec3f>& normals)
{
    nodes->resize(first_frame.size());
    auto voxel_size = kfusion::KinFuParams::default_params_dynamicfusion().volume_size[0] /
                      kfusion::KinFuParams::default_params_dynamicfusion().volume_dims[0];
    for (int i = 0; i < first_frame.size(); i++)
    {
        auto point = first_frame[i];
        auto norm = normals[i];
        if (!std::isnan(point[0]))
        {
            utils::Quaternion<float> t(0.f, point[0], point[1], point[2]);
            utils::Quaternion<float> r(norm);
            nodes->at(i).transform = utils::DualQuaternion<float>(t,r);

            nodes->at(i).vertex = point;
            nodes->at(i).weight = 1; //FIXME: this is just a test
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
                             const std::vector<Vec3f> &live_normals
)
{
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    int i = 0;

    double *parameters = new double[nodes->size() * 6];
    std::vector<cv::Vec3d> double_vertices;
    for(auto v : canonical_vertices)
    {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        if(std::isnan(v[0]))
            continue;

        cv::Vec3f vl(v[0] + 1,v[0] + 2,v[0] + 3);

        ceres::CostFunction* cost_function = DynamicFusionDataEnergy::Create(vl, Vec3f(1,0,0), v, Vec3f(1,0,0), this); // FIXME: send proper parameters, this is a test
        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 parameters);
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
    std::cout << summary.FullReport() << std::endl;

    for(int i = 0; i < nodes->size() * 6; i++)
    {
        std::cout<<parameters[i]<<" ";
        if((i+1) % 6 == 0)
            std::cout<<std::endl;
    }



    float weights[KNN_NEIGHBOURS];
    auto canonical2 = canonical_vertices;
    for(auto v : canonical_vertices)
    {
        utils::Quaternion<float> rotation(0,0,0,0);
        Vec3f translation(0,0,0);
        getWeightsAndUpdateKNN(v, weights);
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
        {
            auto block_position = ret_index[i];
            std::cout<<ret_index[i]<<" Weight:"<<weights[i]<<" ";
            utils::Quaternion<float> rotation1(Vec3f(parameters[block_position],
                                                     parameters[block_position+1],
                                                     parameters[block_position+2]));
            rotation = rotation + weights[i] * rotation1 * nodes->at(block_position).transform.getRotation();

            Vec3f translation1(parameters[block_position+3],
                               parameters[block_position+4],
                               parameters[block_position+5]);
            Vec3f t;
            nodes->at(block_position).transform.getTranslation(t[0],t[1],t[2]);
            translation += weights[i]*t + translation1;
        }
        rotation.rotate(v);
        v += translation;
        std::cout<<std::endl<<"Value of v:"<<v<<std::endl;
    }
    for(auto v : canonical_vertices)
    {
        utils::DualQuaternion<float> final_quat = DQB(v, parameters);
        final_quat.transform(v);
        std::cout<<"Value of v[ "<<i<<" ]:"<<v<<std::endl;
    }
//    for(int i = 0; i < nodes->size() * 6; i+=6)
//    {
//        utils::Quaternion<float> rotation(Vec3f(parameters[i], parameters[i+1], parameters[i+2]));
//        utils::Quaternion<float> translation(0, parameters[i+3], parameters[i+4], parameters[i+5]);
//        auto epsilon = utils::DualQuaternion<float>(translation, rotation);
//        final_quat = final_quat + nodes->at(i / 6).transform * epsilon;
//    }

    delete[] parameters;
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
//    resultSet->init(&ret_index[0], &out_dist_sqr[0]);
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
        cloud.pts[i] = nodes->at(i).vertex;
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