#include "precomp.hpp"
#include "dual_quaternion.hpp"
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>
#include <kfusion/warp_field.hpp>
#include <knn_point_cloud.hpp>
#include <numeric>
#include <opencv/cv.h>
//#include <device.hpp>
#define W_MAX 100.f // This is a hyperparameter for maximum node weight and needs to be tuned. For now set to high value
using namespace kfusion;
using namespace kfusion::cuda;



//void test_points(Intr intr)
//{
//    // Read 3D points
//    std::vector<cv::Point3d> objectPoints;
//
//    std::vector<cv::Point2d> imagePoints;
//    cv::Mat intrisicMat (3,3, cv::DataType<double>::type); // Intrisic matrix
//    intrisicMat.at<double>(0, 0) = intr.fx;
//    intrisicMat.at<double>(1, 0) = 0;
//    intrisicMat.at<double>(2, 0) = intr.cx;
//
//    intrisicMat.at<double>(0, 1) = 0;
//    intrisicMat.at<double>(1, 1) = intr.fy;
//    intrisicMat.at<double>(2, 1) = intr.cy;
//
//    intrisicMat.at<double>(0, 2) = 0;
//    intrisicMat.at<double>(1, 2) = 0;
//    intrisicMat.at<double>(2, 2) = 1;
//
//    cv::Mat rVec(3, 1, cv::DataType<double>::type); // Rotation vector
//    rVec.at<double>(0) = -3.9277902400761393e-002;
//    rVec.at<double>(1) = 3.7803824407602084e-002;
//    rVec.at<double>(2) = 2.6445674487856268e-002;
//
//    cv::Mat tVec(3, 1, cv::DataType<double>::type); // Translation vector
//    tVec.at<double>(0) = 2.1158489381208221e+000;
//    tVec.at<double>(1) = -7.6847683212704716e+000;
//    tVec.at<double>(2) = 2.6169795190294256e+001;
//
//    cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);   // Distortion vector
//    distCoeffs.at<double>(0) = -7.9134632415085826e-001;
//    distCoeffs.at<double>(1) = 1.5623584435644169e+000;
//    distCoeffs.at<double>(2) = -3.3916502741726508e-002;
//    distCoeffs.at<double>(3) = -1.3921577146136694e-002;
//    distCoeffs.at<double>(4) = 1.1430734623697941e+002;
//
//    std::cout << "Intrisic matrix: " << intrisicMat << std::endl << std::endl;
//    std::cout << "Rotation vector: " << rVec << std::endl << std::endl;
//    std::cout << "Translation vector: " << tVec << std::endl << std::endl;
//    std::cout << "Distortion coef: " << distCoeffs << std::endl << std::endl;
//
//    std::vector<cv::Point2d> projectedPoints;
//
//    cv::projectPoints(objectPoints, rVec, tVec, intrisicMat, distCoeffs, projectedPoints);
//    for (unsigned int i = 0; i < projectedPoints.size(); ++i)
//        std::cout << "Image point: " << imagePoints[i] << " Projected to " << projectedPoints[i] << std::endl;
//}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume


kfusion::cuda::TsdfVolume::TsdfVolume(const Vec3i& dims) : data_(),
                                                           trunc_dist_(0.03f),
                                                           max_weight_(128),
                                                           dims_(dims),
                                                           size_(Vec3f::all(3.f)),
                                                           pose_(Affine3f::Identity()),
                                                           gradient_delta_factor_(0.75f),
                                                           raycast_step_factor_(0.75f)
{
    create(dims_);
    cuda::DeviceArray<Point> cloud = fetchCloud(cloud_buffer);
}

kfusion::cuda::TsdfVolume::~TsdfVolume() {}

/**
 * \brief
 * \param dims
 */
void kfusion::cuda::TsdfVolume::create(const Vec3i& dims)
{
    dims_ = dims;
    int voxels_number = dims_[0] * dims_[1] * dims_[2];
    data_.create(voxels_number * sizeof(int));
    setTruncDist(trunc_dist_);
    clear();
}

/**
 * \brief
 * \return
 */
Vec3i kfusion::cuda::TsdfVolume::getDims() const
{
    return dims_;
}

/**
 * \brief
 * \return
 */
Vec3f kfusion::cuda::TsdfVolume::getVoxelSize() const
{
    return Vec3f(size_[0] / dims_[0], size_[1] / dims_[1], size_[2] / dims_[2]);
}

const CudaData kfusion::cuda::TsdfVolume::data() const { return data_; }
CudaData kfusion::cuda::TsdfVolume::data() {  return data_; }
Vec3f kfusion::cuda::TsdfVolume::getSize() const { return size_; }

void kfusion::cuda::TsdfVolume::setSize(const Vec3f& size)
{ size_ = size; setTruncDist(trunc_dist_); }

float kfusion::cuda::TsdfVolume::getTruncDist() const { return trunc_dist_; }

void kfusion::cuda::TsdfVolume::setTruncDist(float distance)
{
    Vec3f vsz = getVoxelSize();
    float max_coeff = std::max<float>(std::max<float>(vsz[0], vsz[1]), vsz[2]);
    trunc_dist_ = std::max (distance, 2.1f * max_coeff);
}

int kfusion::cuda::TsdfVolume::getMaxWeight() const { return max_weight_; }
void kfusion::cuda::TsdfVolume::setMaxWeight(int weight) { max_weight_ = weight; }
Affine3f kfusion::cuda::TsdfVolume::getPose() const  { return pose_; }
void kfusion::cuda::TsdfVolume::setPose(const Affine3f& pose) { pose_ = pose; }
float kfusion::cuda::TsdfVolume::getRaycastStepFactor() const { return raycast_step_factor_; }
void kfusion::cuda::TsdfVolume::setRaycastStepFactor(float factor) { raycast_step_factor_ = factor; }
float kfusion::cuda::TsdfVolume::getGradientDeltaFactor() const { return gradient_delta_factor_; }
void kfusion::cuda::TsdfVolume::setGradientDeltaFactor(float factor) { gradient_delta_factor_ = factor; }
void kfusion::cuda::TsdfVolume::swap(CudaData& data) { data_.swap(data); }
void kfusion::cuda::TsdfVolume::applyAffine(const Affine3f& affine) { pose_ = affine * pose_; }
void kfusion::cuda::TsdfVolume::clear()
{
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::clear_volume(volume);
}

/**
 * \brief
 * \param dists
 * \param camera_pose
 * \param intr
 */
void kfusion::cuda::TsdfVolume::integrate(const Dists& dists, const Affine3f& camera_pose, const Intr& intr)
{
    Affine3f vol2cam = camera_pose.inv() * pose_;

    device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff = device_cast<device::Aff3f>(vol2cam);

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::integrate(dists, volume, aff, proj);
}

/**
 * \brief
 * \param camera_pose
 * \param intr
 * \param depth
 * \param normals
 */
void kfusion::cuda::TsdfVolume::raycast(const Affine3f& camera_pose, const Intr& intr, Depth& depth, Normals& normals)
{
    DeviceArray2D<device::Normal>& n = (DeviceArray2D<device::Normal>&)normals;

    Affine3f cam2vol = pose_.inv() * camera_pose;

    device::Aff3f aff = device_cast<device::Aff3f>(cam2vol);
    device::Mat3f Rinv = device_cast<device::Mat3f>(cam2vol.rotation().inv(cv::DECOMP_SVD));

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::raycast(volume, aff, Rinv, reproj, depth, n, raycast_step_factor_, gradient_delta_factor_);

}

/**
 * \brief
 * \param camera_pose
 * \param intr
 * \param points
 * \param normals
 */
void kfusion::cuda::TsdfVolume::raycast(const Affine3f& camera_pose, const Intr& intr, Cloud& points, Normals& normals)
{
    device::Normals& n = (device::Normals&)normals;
    device::Points& p = (device::Points&)points;

    Affine3f cam2vol = pose_.inv() * camera_pose;

    device::Aff3f aff = device_cast<device::Aff3f>(cam2vol);
    device::Mat3f Rinv = device_cast<device::Mat3f>(cam2vol.rotation().inv(cv::DECOMP_SVD));

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::raycast(volume, aff, Rinv, reproj, p, n, raycast_step_factor_, gradient_delta_factor_);
}

/**
 * \brief
 * \param cloud_buffer
 * \return
 */
DeviceArray<Point> kfusion::cuda::TsdfVolume::fetchCloud(DeviceArray<Point>& cloud_buffer) const
{
    //    enum { DEFAULT_CLOUD_BUFFER_SIZE = 10 * 1000 * 1000 };
    enum { DEFAULT_CLOUD_BUFFER_SIZE = 256 * 256 * 256 };

    if (cloud_buffer.empty ())
        cloud_buffer.create (DEFAULT_CLOUD_BUFFER_SIZE);

    DeviceArray<device::Point>& b = (DeviceArray<device::Point>&)cloud_buffer;

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff  = device_cast<device::Aff3f>(pose_);

    device::TsdfVolume volume((ushort2*)data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    size_t size = extractCloud(volume, aff, b);

    return DeviceArray<Point>((Point*)cloud_buffer.ptr(), size);
}


void kfusion::cuda::TsdfVolume::fetchNormals(const DeviceArray<Point>& cloud, DeviceArray<Normal>& normals) const
{
    normals.create(cloud.size());
    DeviceArray<device::Point>& c = (DeviceArray<device::Point>&)cloud;

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff  = device_cast<device::Aff3f>(pose_);
    device::Mat3f Rinv = device_cast<device::Mat3f>(pose_.rotation().inv(cv::DECOMP_SVD));

    device::TsdfVolume volume((ushort2*)data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::extractNormals(volume, c, aff, Rinv, gradient_delta_factor_, (float4*)normals.ptr());
}

/**
 * \brief
 * \param vertex
 * \param voxel_center
 * \param weight
 */
void kfusion::cuda::TsdfVolume::compute_tsdf_value(Vec3f vertex, Vec3f voxel_center, float weight)
{
    float new_weight;
    for(auto entry : tsdf_entries)
    {
        float w_x = std::min(entry.tsdf_weight + new_weight, W_MAX);
        float ro; // = psdf(stuff)
        float v_x = entry.tsdf_value * tsdf_entries[0].tsdf_weight + std::min(ro, trunc_dist_) * w_x;
        v_x = v_x / (entry.tsdf_weight + w_x);
        entry.tsdf_value = v_x;
        entry.tsdf_weight = w_x;
    }
}

/**
 * \brief
 * \param warp_field
 * \param depth_img
 * \param camera_pose
 * \param intr
 */
void kfusion::cuda::TsdfVolume::surface_fusion(const WarpField& warp_field,
                                               const cuda::Dists& depth_img,
                                               const Affine3f& camera_pose,
                                               const Intr& intr)
{
    const std::vector<deformation_node> *nodes = warp_field.getNodes();
    utils::PointCloud cloud;
    cloud.pts.resize(nodes->size());

    for (size_t i = 0; i < nodes->size(); i++)
        (*nodes)[i].transform.getTranslation(cloud.pts[i]);

    kd_tree_t index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();
    const size_t k = 8; //FIXME: number of neighbours should be a hyperparameter
    std::vector<utils::DualQuaternion<float>> neighbours(k);
    std::vector<size_t> ret_index(k);
    std::vector<float> out_dist_sqr(k);
    nanoflann::KNNResultSet<float> resultSet(k);
    resultSet.init(&ret_index[0], &out_dist_sqr[0]);
//
//
//    cuda::DeviceArray<Point> points = fetchCloud(cloud_buffer);
    std::vector<Point, std::allocator<Point>> cloud_host(cloud_buffer.size());
//    cloud_buffer.download(cloud_host);
    //    cloud_host.create(1, (int)cloud.size(), CV_32FC4);
    //    cloud.download(cloud_host.ptr<Point>());
    //    std::vector<Point> cloud_host_vector(cloud_host);

    std::vector<Point, std::allocator<Point>> cloud_initial(cloud_host);
//
//    fetchNormals(cloud_buffer, normal_buffer);
//    std::vector<Point, std::allocator<Point>> normals_host(cloud_buffer.size());
//    normal_buffer.download(normals_host);
//    std::vector<Point, std::allocator<Point>> normals_initial(normals_host);
//
//    warp_field.warp(cloud_host, normals_host);

    //    assert(tsdf_entries.size() == cloud_host.size() == normals_host.size());
    for(size_t i = 0; i < cloud_initial.size(); i++)
    {
        Vec3f initial(cloud_initial[i].x,cloud_initial[i].y,cloud_initial[i].z);
        Vec3f warped(cloud_host[i].x,cloud_host[i].y,cloud_host[i].z);
        float ro = psdf(initial, warped, depth_img, intr);
//
        if(ro > -trunc_dist_)
        {
            index.findNeighbors(resultSet, warped.val, nanoflann::SearchParams(10));
            float weight = weighting(out_dist_sqr, k);
            float coeff = std::min(ro, trunc_dist_);

            tsdf_entries[i].tsdf_value = tsdf_entries[i].tsdf_value * tsdf_entries[i].tsdf_weight + coeff * weight;
            tsdf_entries[i].tsdf_value = tsdf_entries[i].tsdf_weight + weight;

            tsdf_entries[i].tsdf_weight = std::min(tsdf_entries[i].tsdf_weight + weight, W_MAX);
        }
//                else
        //        stays the same
    }
}
//FIXME: docstring is not up to date
//TODO:
/**
 * \fn TSDF::psdf (Mat3f K, Depth& depth, Vec3f voxel_center)
 * \brief return a quaternion that is the spherical linear interpolation between q1 and q2
 *        where percentage (from 0 to 1) defines the amount of interpolation
 * \param K: camera matrix
 * \param depth: a depth frame
 * \param voxel_center
 *
 */
float kfusion::cuda::TsdfVolume::psdf(Vec3f voxel_center,
                                      Vec3f warped,
                                      const Dists& depth_img,
                                      const Intr& intr)
{
    device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);
    float3 point = make_float3(warped[0], warped[1], warped[2]);
    device::project(depth_img, point, proj);
    //    return (K.inv() * depth.(u_c[0], u_c[1])*[u_c.T, 1].T).z - x_t.z;
    return 0;
}

/**
 * \brief
 * \param dist_sqr
 * \param k
 * \return
 */
float kfusion::cuda::TsdfVolume::weighting(const std::vector<float>& dist_sqr, int k) const
{
    float distances = 0;
    for(auto distance : dist_sqr)
        distances += sqrt(distance);
    return distances / k;
}
