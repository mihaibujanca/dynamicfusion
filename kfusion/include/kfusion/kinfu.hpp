#pragma once

#include <kfusion/types.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/cuda/projective_icp.hpp>
#include <vector>
#include <string>
#include <dual_quaternion.hpp>
#include <quaternion.hpp>

namespace kfusion
{
    namespace cuda
    {
        KF_EXPORTS int getCudaEnabledDeviceCount();
        KF_EXPORTS void setDevice(int device);
        KF_EXPORTS std::string getDeviceName(int device);
        KF_EXPORTS bool checkIfPreFermiGPU(int device);
        KF_EXPORTS void printCudaDeviceInfo(int device);
        KF_EXPORTS void printShortCudaDeviceInfo(int device);
    }

//  TODO: Adapt this and nanoflann to work with Quaternions. Probably needs an adaptor class
// Check https://github.com/jlblancoc/nanoflann/blob/master/examples/pointcloud_adaptor_example.cpp
    template <typename T>
    struct PointCloud
    {
        struct Point
        {
            T  x,y,z;
            public:
                inline Point(T x,T y,T z) : x(x), y(y), z(z){}
                inline Point() : x(0), y(0), z(0){}
        };

        std::vector<Point> pts;

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const { return pts.size(); }

        // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
        inline T kdtree_distance(const Point p1, const size_t idx_p2,size_t /*size*/) const
        {
            const T d0=p1.x-pts[idx_p2].x;
            const T d1=p1.y-pts[idx_p2].y;
            const T d2=p1.z-pts[idx_p2].z;
            return d0*d0+d1*d1+d2*d2;
        }

  // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
        inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t /*size*/) const
        {
            const T d0=p1[0]-pts[idx_p2].x;
            const T d1=p1[1]-pts[idx_p2].y;
            const T d2=p1[2]-pts[idx_p2].z;
            return d0*d0+d1*d1+d2*d2;
        }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline T kdtree_get_pt(const size_t idx, int dim) const
        {
            if (dim==0) return pts[idx].x;
            else if (dim==1) return pts[idx].y;
            else return pts[idx].z;
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

    };
    struct KF_EXPORTS KinFuParams
    {
        static KinFuParams default_params();

        int cols;  //pixels
        int rows;  //pixels

        Intr intr;  //Camera parameters

        Vec3i volume_dims; //number of voxels
        Vec3f volume_size; //meters
        Affine3f volume_pose; //meters, inital pose

        float bilateral_sigma_depth;   //meters
        float bilateral_sigma_spatial;   //pixels
        int   bilateral_kernel_size;   //pixels

        float icp_truncate_depth_dist; //meters
        float icp_dist_thres;          //meters
        float icp_angle_thres;         //radians
        std::vector<int> icp_iter_num; //iterations for level index 0,1,..,3

        float tsdf_min_camera_movement; //meters, integrate only if exceedes
        float tsdf_trunc_dist;             //meters;
        int tsdf_max_weight;               //frames

        float raycast_step_factor;   // in voxel sizes
        float gradient_delta_factor; // in voxel sizes

        Vec3f light_pose; //meters

    };

    class KF_EXPORTS KinFu
    {
    public:        
        typedef cv::Ptr<KinFu> Ptr;

        KinFu(const KinFuParams& params);

        const KinFuParams& params() const;
        KinFuParams& params();

        const cuda::TsdfVolume& tsdf() const;
        cuda::TsdfVolume& tsdf();

        const cuda::ProjectiveICP& icp() const;
        cuda::ProjectiveICP& icp();

        void reset();

        bool operator()(const cuda::Depth& depth, const cuda::Image& image = cuda::Image());

        void renderImage(cuda::Image& image, int flags = 0);
        void renderImage(cuda::Image& image, const Affine3f& pose, int flags = 0);
        utils::DualQuaternion<float> DQB(Vec3f vertex,
                                          std::vector<utils::DualQuaternion<float>> nodes,
                                          float voxel_size);
//        std::pair<Vec3f,Vec3f>
        std::vector<utils::DualQuaternion<float>> warp(std::vector<Vec3f>& frame,
                                                       const cuda::TsdfVolume& tsdfVolume);
        float weighting(Vec3f vertex, Vec3f voxel_center, float weight);

        Affine3f getCameraPose (int time = -1) const;
    private:
        void allocate_buffers();

        int frame_counter_;
        KinFuParams params_;

        std::vector<Affine3f> poses_;

        cuda::Dists dists_;
        cuda::Frame curr_, prev_;

        cuda::Cloud points_;
        cuda::Normals normals_;
        cuda::Depth depths_;

        cv::Ptr<cuda::TsdfVolume> volume_;
        cv::Ptr<cuda::ProjectiveICP> icp_;
    };
}
