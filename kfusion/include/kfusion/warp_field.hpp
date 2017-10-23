#ifndef KFUSION_WARP_FIELD_HPP
#define KFUSION_WARP_FIELD_HPP

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


    /*!
     * \struct node
     * \brief A node of the warp field
     * \details The state of the warp field Wt at time t is defined by the values of a set of n
     * deformation nodes Nt_warp = {dg_v, dg_w, dg_se3}_t. Here, this is represented as follows
     *
     * \var node::vertex
     * Position of the vertex in space. This will be used when computing KNN for warping points.
     *
     * \var node::transform
     * Transformation for each vertex to warp it into the live frame, equivalent to dg_se in the paper.
     *
     * \var node::weight
     * Equivalent to dg_w
     */
    struct deformation_node
    {
        Vec3f vertex;
        kfusion::utils::DualQuaternion<float> transform;
        float weight = 0;
    };
    class WarpField
    {
    public:
        WarpField();
        ~WarpField();

        void init(const cv::Mat& first_frame);
        void init(const std::vector<Vec3f>& first_frame);
        void energy(const cuda::Cloud &frame,
                    const cuda::Normals &normals,
                    const Affine3f &pose,
                    const cuda::TsdfVolume &tsdfVolume,
                    const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
                            kfusion::utils::DualQuaternion<float>>> &edges
        );

        void energy_data(const std::vector<Vec3f> &canonical_vertices,
                          const std::vector<Vec3f> &canonical_normals,
                          const std::vector<Vec3f> &live_vertices,
                          const std::vector<Vec3f> &live_normals);
        void energy_reg(const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
                kfusion::utils::DualQuaternion<float>>> &edges);


        void warp(std::vector<Vec3f>& points, std::vector<Vec3f>& normals) const;
        void warp(cuda::Cloud& points) const;

        utils::DualQuaternion<float> DQB(const Vec3f& vertex) const;
        utils::DualQuaternion<float> DQB(const Vec3f& vertex, const std::vector<double*> epsilon) const;
        void update_nodes(const double *epsilon);

        void getWeightsAndUpdateKNN(const Vec3f& vertex, float weights[KNN_NEIGHBOURS]) const;

        float weighting(float squared_dist, float weight) const;
        void KNN(Vec3f point) const;

        void clear();

        const std::vector<deformation_node>* getNodes() const;
        std::vector<deformation_node>* getNodes();
        const cv::Mat getNodesAsMat() const;
        void setWarpToLive(const Affine3f &pose);
        std::vector<float>* getDistSquared() const;
        std::vector<size_t>* getRetIndex() const;

    private:
        std::vector<deformation_node>* nodes_;
        kd_tree_t* index_;
        Affine3f warp_to_live_;
        void buildKDTree();
    };
}
#endif //KFUSION_WARP_FIELD_HPP
