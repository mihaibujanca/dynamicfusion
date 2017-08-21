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
        break;
        auto point = cloud_host[i];
        auto norm = normals_host[i];
        if(!std::isnan(point.x))
        {

        }
        else
        {
//            std::cout<<"NANS"<<std::endl;
//            break;
        }
    }

//    if (m_pWarpField->getNumNodesInLevel(0) == 0)
//    {
//        printf("no warp nodes, return\n");
//        return FLT_MAX;
//    }
//
//    m_vmap_warp = &vmap_warp;
//    m_nmap_warp = &nmap_warp;
//    m_vmap_live = &vmap_live;
//    m_nmap_live = &nmap_live;
//
//    // perform Gauss-Newton iteration
//    //for (int k = 0; k < 100; k++)
//    float totalEnergy = 0.f, data_energy=0.f, reg_energy=0.f;
//    m_pWarpField->extract_nodes_info_no_allocation(m_nodesKnn, m_twist, m_nodesVw);
//    for (int iter = 0; iter < m_param->fusion_GaussNewton_maxIter; iter++)
//    {
//
//        m_Hd = 0.f;
//        cudaSafeCall(cudaMemset(m_g.ptr(), 0, sizeof(float)*m_g.size()),
//                     "GpuGaussNewtonSolver::solve, setg=0");
//
//        checkNan(m_twist, m_numNodes, ("twist_" + std::to_string(iter)).c_str());
//
//        // 1. calculate data term: Hd += Jd'Jd; g += Jd'fd
//        // 2. calculate reg term: Jr = [Jr0 Jr1; 0 Jr3]; fr;
//        // 3. calculate Hessian: Hd += Jr0'Jr0; B = Jr0'Jr1; Hr = Jr1'Jr1 + Jr3'Jr3; g=-(g+Jr'*fr)
//        // 4. solve H*h = g
//        if (m_param->graph_single_level)
//            singleLevelSolve();
//        else
//            blockSolve()
//        // if not fix step, we perform line search
//        if (m_param->fusion_GaussNewton_fixedStep <= 0.f)
//        {
//            float old_energy = calcTotalEnergy(data_energy, reg_energy);
//            float new_energy = 0.f;
//            float alpha = 1.f;
//            const static float alpha_stop = 1e-2;
//            cudaSafeCall(cudaMemcpy(m_tmpvec.ptr(), m_twist.ptr(), m_Jr->cols()*sizeof(float),
//                                    cudaMemcpyDeviceToDevice), "copy tmp vec to twist");
//            for (; alpha > alpha_stop; alpha *= 0.5)
//            {
//                // x += alpha * h
//                updateTwist_inch(m_h.ptr(), alpha);
//                new_energy = calcTotalEnergy(data_energy, reg_energy);
//                if (new_energy < old_energy)
//                    break;
//                // reset x
//                cudaSafeCall(cudaMemcpy(m_twist.ptr(), m_tmpvec.ptr(),
//                                        m_Jr->cols()*sizeof(float), cudaMemcpyDeviceToDevice), "copy twist to tmp vec");
//            }
//            totalEnergy = new_energy;
//            if (alpha <= alpha_stop)
//                break;
//            float norm_h = 0.f, norm_g = 0.f;
//            cublasStatus_t st = cublasSnrm2(m_cublasHandle, m_Jr->cols(),
//                                            m_h.ptr(), 1, &norm_h);
//            st = cublasSnrm2(m_cublasHandle, m_Jr->cols(),
//                             m_g.ptr(), 1, &norm_g);
//            if (norm_h < (norm_g + 1e-6f) * 1e-6f)
//                break;
//        }
//            // else, we perform fixed step update.
//        else
//        {
//            // 5. accumulate: x += step * h;
//            updateTwist_inch(m_h.ptr(), m_param->fusion_GaussNewton_fixedStep);
//        }
//    }// end for iter
//
//    if (m_param->fusion_GaussNewton_fixedStep > 0.f)
//        totalEnergy = calcTotalEnergy(data_energy, reg_energy);
//
//    if (data_energy_)
//        *data_energy_ = data_energy;
//    if (reg_energy_)
//        *reg_energy_ = reg_energy;
//
//    return totalEnergy;
//}
}

/**
 * \brief
 * \param frame
 * \param pose
 * \param tsdfVolume
 */
void WarpField::energy_data(Vec3f v,
                            Vec3f n
)
{
    if (std::isnan(n[0]) || std::isnan(v[0]))
        return;
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
//    const int x = threadIdx.x + blockIdx.x * blockDim.x;
//    const int y = threadIdx.y + blockIdx.y * blockDim.y;
//    if (x >= imgWidth || y >= imgHeight)
//        return;
//
//    const KnnIdx knn = vmapKnn(y, x);
//    Tbx::Point3 v(convert(read_float3_4(vmap_cano(y, x))));
//    Tbx::Vec3 n(convert(read_float3_4(nmap_cano(y, x))));
//
//    if (isnan(n.x) || isnan(v.x))
//        return;
//
//    // 1. get all nodes params
//    // 2. compute function
//    float wk[KnnK];
//    auto dq = calc_pixel_dq(knn, v, wk);
//    float norm_dq = dq.norm();
//    if (norm_dq < Tbx::Dual_quat_cu::epsilon())
//        return;
//    auto dq_not_normalized = dq;
//    dq = dq * (1.f / norm_dq); // normalize
//
//    // compute vl
//
//    // the grad energy
//    const float f = nwarp.dot(vwarp - vl); //normal * (v-vl)
//    const float psi_f = data_term_penalty(f);
//
//    // 3. compute jacobi
//    for (int knnK = 0; knnK < KnnK; knnK++)
//    {
//        if (knn_k(knn, knnK) >= nNodes)
//            break;
//        float df[6];
//
//        // 3.0 p_rotation[0:2]
//        for (int i = 0; i < 3; i++)
//        {
//            float inc;
//            Tbx::Dual_quat_cu dq1 = dq_not_normalized;
//            exchange_ri_k(knn, wk, knnK, i, dq1, inc);
//            dq1 *= (1.f / dq1.norm());
//            nwarp = Tlw*dq1.rotate(n);
//            vwarp = Tlw*dq1.transform(v);
//
//            float f1 = nwarp.dot(vwarp - vl);
//            df[i] = (f1 - f) / inc;
//        }// i=0:3
//
//        // 3.1 p_translation[0:2]
//        for (int i = 0; i < 3; i++)
//        {
//            float inc;
//            Tbx::Dual_quat_cu dq1 = dq_not_normalized;
//            exchange_ti_k(knn, wk, knnK, i, dq1, inc);
//            dq1 *= (1.f / dq1.norm());
//            nwarp = Tlw*dq1.rotate(n);
//            vwarp = Tlw*dq1.transform(v);
//
//            float f1 = nwarp.dot(vwarp - vl);
//            df[i+3] = (f1 - f) / inc;
//        }// i=0:3
//
//        //// reduce--------------------------------------------------
//        int shift = knn_k(knn, knnK) * VarPerNode2;
//        int shift_g = knn_k(knn, knnK) * VarPerNode;
//        for (int i = 0; i < VarPerNode; ++i)
//        {
//            for (int j = 0; j <= i; ++j)
//                atomicAdd(&Hd_[shift + j], df[i] * df[j]);
//            atomicAdd(&g_[shift_g + i], df[i] * psi_f);
//            shift += VarPerNode;
//        }
//    }
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