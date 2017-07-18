#include "precomp.hpp"
#include "dual_quaternion.hpp"
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>
#include <kfusion/warp_field.hpp>

#define W_MAX 100.f // This is a hyperparameter for maximum node weight and needs to be tuned. For now set to high value
using namespace kfusion;
using namespace kfusion::cuda;

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
}

kfusion::cuda::TsdfVolume::~TsdfVolume() {}

void kfusion::cuda::TsdfVolume::create(const Vec3i& dims)
{
    dims_ = dims;
    int voxels_number = dims_[0] * dims_[1] * dims_[2];
    data_.create(voxels_number * sizeof(int));
    setTruncDist(trunc_dist_);
    clear();
}

Vec3i kfusion::cuda::TsdfVolume::getDims() const
{ return dims_; }

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

void kfusion::cuda::TsdfVolume::surface_fusion(const WarpField& warp_field,
                                               const cuda::Depth& depth_img,
                                               const Affine3f& camera_pose,
                                               const Intr& intr)
{

    cuda::DeviceArray<Point> cloud_buffer;
    cuda::DeviceArray<Point> cloud = fetchCloud(cloud_buffer);
    std::vector<Point, std::allocator<Point>> cloud_host(cloud_buffer.size());
    cloud_buffer.download(cloud_host);
    std::vector<Point, std::allocator<Point>> cloud_initial(cloud_host);

    cuda::DeviceArray<Normal> normal_buffer;
    fetchNormals(cloud_buffer, normal_buffer);
    std::vector<Point, std::allocator<Point>> normals_host(cloud_buffer.size());
    normal_buffer.download(normals_host);
    std::vector<Point, std::allocator<Point>> normals_initial(normals_host);

    warp_field.warp(cloud_host, normals_host);

//    assert(tsdf_entries.size() == cloud_host.size() == normals_host.size());
    for(size_t i = 0; i < cloud_initial.size(); i++)
    {
        Vec3f initial(cloud_initial[i].x,cloud_initial[i].y,cloud_initial[i].z);
        Vec3f warped(cloud_host[i].x,cloud_host[i].y,cloud_host[i].z);
        float ro = psdf(initial, warped, depth_img, intr);

        if(ro > -trunc_dist_)
        {
            float weight = weighting(initial);
            float coeff = std::min(ro, trunc_dist_);

            tsdf_entries[i].tsdf_value = tsdf_entries[i].tsdf_value * tsdf_entries[i].tsdf_weight + coeff * weight;
            tsdf_entries[i].tsdf_value = tsdf_entries[i].tsdf_weight + weight;

            tsdf_entries[i].tsdf_weight = std::min(tsdf_entries[i].tsdf_weight + weight, W_MAX);
        }
//        else stays the same
    }
}

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
                                      const Depth& depth_img,
                                      const Intr& intr)
{
    cv::Vec3f u_c;
    Mat3f K(intr.fx, 0, intr.cx, 0, intr.fy, intr.cy, 0, 0, 1);
    cv::perspectiveTransform(warped, u_c, K);
    //    auto u_c_4 = (u_c.t()., 1);
    //    return (K.inv() * depth.(u_c[0], u_c[1])*[u_c.T, 1].T).z - x_t.z;
    return 0;
}


float kfusion::cuda::TsdfVolume::weighting(Vec3f voxel_center)
{
    return 0;
}