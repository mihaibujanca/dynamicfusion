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
nanoflann::KNNResultSet<float> *resultSet_;
std::vector<float> out_dist_sqr_;
std::vector<size_t> ret_index_;

WarpField::WarpField()
{
    nodes_ = new std::vector<deformation_node>();
    index_ = new kd_tree_t(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    ret_index_ = std::vector<size_t>(KNN_NEIGHBOURS);
    out_dist_sqr_ = std::vector<float>(KNN_NEIGHBOURS);
    resultSet_ = new nanoflann::KNNResultSet<float>(KNN_NEIGHBOURS);
    resultSet_->init(&ret_index_[0], &out_dist_sqr_[0]);
    neighbours = std::vector<utils::DualQuaternion<float>>(KNN_NEIGHBOURS);

}

WarpField::~WarpField()
{
    delete[] nodes_;
    delete resultSet_;
    delete index_;
}

/**
 *
 * @param first_frame
 * @param normals
 */
void WarpField::init(const cv::Mat& first_frame, const cv::Mat& normals)
{
    assert(first_frame.rows == normals.rows);
    assert(first_frame.cols == normals.cols);
    nodes_->resize(first_frame.cols * first_frame.rows);
    auto voxel_size = kfusion::KinFuParams::default_params_dynamicfusion().volume_size[0] /
                      kfusion::KinFuParams::default_params_dynamicfusion().volume_dims[0];

//    FIXME:: this is a test, remove later
    voxel_size = 1;
    for(int i = 0; i < first_frame.rows; i++)
        for(int j = 0; j < first_frame.cols; j++)
        {
            auto point = first_frame.at<Point>(i,j);
            auto norm = normals.at<Normal>(i,j);
            if(!std::isnan(point.x))
            {
                utils::Quaternion<float> r(Vec3f(norm.x,norm.y,norm.z));
                if(std::isnan(r.w_) || std::isnan(r.x_) ||std::isnan(r.y_) ||std::isnan(r.z_))
                    continue;

                utils::Quaternion<float> t(0,point.x, point.y, point.z);
                nodes_->at(i*first_frame.cols+j).transform = utils::DualQuaternion<float>(t, r);

                nodes_->at(i*first_frame.cols+j).vertex = Vec3f(point.x,point.y,point.z);
                nodes_->at(i*first_frame.cols+j).weight = 3 * voxel_size;
            }
        }
    buildKDTree();
}

/**
 *
 * @param first_frame
 * @param normals
 */
void WarpField::init(const std::vector<Vec3f>& first_frame, const std::vector<Vec3f>& normals)
{
    nodes_->resize(first_frame.size());
    auto voxel_size = kfusion::KinFuParams::default_params_dynamicfusion().volume_size[0] /
                      kfusion::KinFuParams::default_params_dynamicfusion().volume_dims[0];

//    FIXME: this is a test, remove
    voxel_size = 1;
    for (int i = 0; i < first_frame.size(); i++)
    {
        auto point = first_frame[i];
        auto norm = normals[i];
        if (!std::isnan(point[0]))
        {
            utils::Quaternion<float> t(0.f, point[0], point[1], point[2]);
            utils::Quaternion<float> r(norm);
            nodes_->at(i).transform = utils::DualQuaternion<float>(t,r);

            nodes_->at(i).vertex = point;
            nodes_->at(i).weight = 3 * voxel_size;
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
 *
 * @param canonical_vertices
 * @param canonical_normals
 * @param live_vertices
 * @param live_normals
 * @return
 */
float WarpField::energy_data(const std::vector<Vec3f> &canonical_vertices,
                             const std::vector<Vec3f> &canonical_normals,
                             const std::vector<Vec3f> &live_vertices,
                             const std::vector<Vec3f> &live_normals
)
{

//    assert((canonical_normals.size() == canonical_vertices.size()) == (live_normals.size() == live_vertices.size()));
    ceres::Problem problem;
    std::vector<cv::Vec3d> double_vertices;
    float weights[KNN_NEIGHBOURS];
    unsigned long indices[KNN_NEIGHBOURS];

    WarpProblem warpProblem(this);
    std::vector<double*> params;
    for(int i = 0; i < live_vertices.size(); i++)
    {
        if(std::isnan(canonical_vertices[i][0]))
            continue;
        getWeightsAndUpdateKNN(canonical_vertices[i], weights);

        for(int j = 0; j < KNN_NEIGHBOURS; j++)
        {
            indices[j] = ret_index_[j];
            std::cout<<"Weight["<<j<<"]="<<weights[j]<<" ";
        }
        std::cout<<std::endl;

        params = warpProblem.mutable_epsilon(indices);
        ceres::CostFunction* cost_function = DynamicFusionDataEnergy::Create(live_vertices[i],
                                                                             live_normals[i],
                                                                             canonical_vertices[i],
                                                                             canonical_normals[i],
                                                                             this,
                                                                             weights,
                                                                             indices);
        problem.AddResidualBlock(cost_function,  NULL /* squared loss */, params);

    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;


    auto all_params = warpProblem.params();
    for(int i = 0; i < nodes_->size() * 6; i++)
    {
        std::cout<<all_params[i]<<" ";
        if((i+1) % 6 == 0)
            std::cout<<std::endl;
    }

    for(auto v : canonical_vertices)
    {
        utils::Quaternion<float> rotation(0,0,0,0);
        Vec3f translation(0,0,0);
        getWeightsAndUpdateKNN(v, weights);
        params = warpProblem.mutable_epsilon(ret_index_);
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
        {

            Vec3f translation1(params[i][3],
                               params[i][4],
                               params[i][5]);

            Vec3f dq_translation;
            nodes_->at(ret_index_[i]).transform.getTranslation(dq_translation);

            translation = translation1 + dq_translation;
            translation *= weights[i];
            v += translation;
        }

        std::cout<<std::endl<<"Value of v:"<<v;
    }
    std::cout<<std::endl;
    for(auto v : canonical_vertices)
    {
        KNN(v);
        Vec3f v1 = v, test;
        params = warpProblem.mutable_epsilon(ret_index_);
        auto dq = DQB(v, params);
//        dq.transform(v);
//        std::cout<<"FROM DQB:"<<v<<std::endl;

        dq.getTranslation(test);
        std::cout<<"Translation only:"<<test+v1<<std::endl;
    }
    exit(0);
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
 *
 * @param points
 * @param normals
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
        point = warp_to_live_ * point;

        dqb.transform(normals[i]);
        normals[i] = warp_to_live_ * normals[i];
        i++;
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
        quaternion_sum = quaternion_sum + weighting(out_dist_sqr_[i], nodes_->at(ret_index_[i]).weight) *
                                          nodes_->at(ret_index_[i]).transform;

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
utils::DualQuaternion<float> WarpField::DQB(const Vec3f& vertex, const std::vector<double*> epsilon) const
{
    float weights[KNN_NEIGHBOURS];
    getWeightsAndUpdateKNN(vertex, weights);
    utils::DualQuaternion<float> quaternion_sum;
    utils::DualQuaternion<float> eps;
    utils::Quaternion<float> translation_sum(0,0,0,0);
    utils::Quaternion<float> rotation(0,0,0,0);
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
    {
        // epsilon [0:2] is rotation [3:5] is translation
//        eps.from_twist(epsilon[i][0],epsilon[i][1],epsilon[i][2],epsilon[i][3],epsilon[i][4],epsilon[i][5]);
        utils::Quaternion<float> translation1(0, epsilon[i][3],epsilon[i][4],epsilon[i][5]);
        translation_sum = translation_sum + weights[i] * (nodes_->at(ret_index_[i]).transform.getTranslation() + translation1);

    }

    auto norm = quaternion_sum.magnitude();

//    return utils::DualQuaternion<float>(quaternion_sum.getRotation() / norm.first,
//                                        quaternion_sum.getTranslation() / norm.second);
    return utils::DualQuaternion<float>(utils::Quaternion<float>(),
                                        translation_sum);
}

/**
 * \brief
 * \param vertex
 * \param weight
 * \return
 */
void WarpField::getWeightsAndUpdateKNN(const Vec3f& vertex, float weights[KNN_NEIGHBOURS]) const
{
    KNN(vertex);
    for (size_t i = 0; i < KNN_NEIGHBOURS; i++)
        weights[i] = weighting(out_dist_sqr_[i], nodes_->at(ret_index_[i]).weight);
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
    resultSet_->init(&ret_index_[0], &out_dist_sqr_[0]);
    index_->findNeighbors(*resultSet_, point.val, nanoflann::SearchParams(10));
}

/**
 * \brief
 * \return
 */
const std::vector<deformation_node>* WarpField::getNodes() const
{
    return nodes_;
}

/**
 * \brief
 * \return
 */
void WarpField::buildKDTree()
{
    //    Build kd-tree with current warp nodes.
    cloud.pts.resize(nodes_->size());
    for(size_t i = 0; i < nodes_->size(); i++)
        cloud.pts[i] = nodes_->at(i).vertex;
    index_->buildIndex();
}

const cv::Mat WarpField::getNodesAsMat() const
{
    cv::Mat matrix(1, nodes_->size(), CV_32FC3);
    for(int i = 0; i < nodes_->size(); i++)
        matrix.at<cv::Vec3f>(i) = nodes_->at(i).vertex;
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
    warp_to_live_ = pose;
}

std::vector<float>* WarpField::getDistSquared() const
{
    return &out_dist_sqr_;
}