#ifndef KFUSION_WARP_FIELD_HPP
#define KFUSION_WARP_FIELD_HPP

/**
 * \brief
 * \details
 */
#include <dual_quaternion.hpp>
#include <kfusion/types.hpp>
#include <nanoflann/nanoflann.hpp>
#include <knn_point_cloud.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#define KNN_NEIGHBOURS 8

namespace kfusion
{
    typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, utils::PointCloud>,
            utils::PointCloud,
            3 /* dim */
    > kd_tree_t;


    //    TODO: remember to rewrite this with proper doxygen formatting (e.g <sub></sub> rather than _
    /*!
     * \struct node
     * \brief A node of the warp field
     * \details The state of the warp field Wt at time t is defined by the values of a set of n
     * deformation nodes Nt_warp = {dg_v, dg_w, dg_se3}_t. Here, this is represented as follows
     *
     * \var node::index
     * Index of the node in the canonical frame. Equivalent to dg_v
     *
     * \var node::transform
     * Transform from canonical point to warped point, equivalent to dg_se in the paper.
     *
     * \var node::weight
     * Equivalent to dg_w
     */
    struct deformation_node
    {
        Vec3f vertex;
        kfusion::utils::DualQuaternion<float> transform;
        float weight = 0;
        bool valid = true;
    };
    class WarpField
    {
    public:
        WarpField();
        ~WarpField();

        void init(const cv::Mat& first_frame, const cv::Mat& normals);
        void energy(const cuda::Cloud &frame,
                    const cuda::Normals &normals,
                    const Affine3f &pose,
                    const cuda::TsdfVolume &tsdfVolume,
                    const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
                            kfusion::utils::DualQuaternion<float>>> &edges
        );
        void energy_temp(const Affine3f &pose);

        void energy_data(const cuda::Depth &frame,
                         const Affine3f &pose,
                         const cuda::TsdfVolume &tsdfVolume
        );

        void energy_reg(const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
                kfusion::utils::DualQuaternion<float>>> &edges);

        float tukeyPenalty(float x, float c) const;

        float huberPenalty(float a, float delta) const;

        void warp(std::vector<Point, std::allocator<Point>>& cloud_host,
                  std::vector<Point, std::allocator<Point>>& normals_host) const;

        void warp(std::vector<Vec3f>& points) const;

        utils::DualQuaternion<float> DQB(const Vec3f& vertex) const;

        float weighting(Vec3f vertex, Vec3f voxel_center, float weight) const;
        float weighting(float squared_dist, float weight) const;
        void KNN(Vec3f point) const;

        //        std::vector<kfusion::utils::DualQuaternion<float>> getQuaternions() const;
        void clear();

        const std::vector<deformation_node>* getNodes() const;
        const cv::Mat getNodesAsMat() const;
        std::vector<float> out_dist_sqr; //FIXME: shouldn't be public

    private:
        //    FIXME: should be a pointer
        std::vector<deformation_node> nodes;
        kd_tree_t* index;
        std::vector<size_t> ret_index;
        nanoflann::KNNResultSet<float> *resultSet;
    };
}
#endif //KFUSION_WARP_FIELD_HPP
