#pragma once

#include <kfusion/types.hpp>

namespace kfusion
{
    namespace cuda
    {
        class KF_EXPORTS TsdfVolume
        {
        public:
            TsdfVolume(const cv::Vec3i& dims);
            virtual ~TsdfVolume();

            void create(const Vec3i& dims);

            Vec3i getDims() const;
            Vec3f getVoxelSize() const;

            const CudaData data() const;
            CudaData data();

            Vec3f getSize() const;
            void setSize(const Vec3f& size);

            float getTruncDist() const;
            void setTruncDist(float distance);

            int getMaxWeight() const;
            void setMaxWeight(int weight);

            Affine3f getPose() const;
            void setPose(const Affine3f& pose);

            float getRaycastStepFactor() const;
            void setRaycastStepFactor(float factor);

            float getGradientDeltaFactor() const;
            void setGradientDeltaFactor(float factor);

            Vec3i getGridOrigin() const;
            void setGridOrigin(const Vec3i& origin);

            virtual void clear();
            virtual void applyAffine(const Affine3f& affine);
            virtual void integrate(const Dists& dists, const Affine3f& camera_pose, const Intr& intr);
            virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Depth& depth, Normals& normals);
            virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Cloud& points, Normals& normals);

            void swap(CudaData& data);

            DeviceArray<Point> fetchCloud(DeviceArray<Point>& cloud_buffer) const;
            void fetchNormals(const DeviceArray<Point>& cloud, DeviceArray<Normal>& normals) const;

            struct Entry
            {
                typedef unsigned short half;

                half tsdf;
                unsigned short weight;

                static float half2float(half value);
                static half float2half(float value);
            };
        private:
            CudaData data_;

            float trunc_dist_;
            int max_weight_;
            Vec3i dims_;
            Vec3f size_;
            Affine3f pose_;

            float gradient_delta_factor_;
            float raycast_step_factor_;
        };
    }
}
