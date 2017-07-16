#ifndef KFUSION_WARP_FIELD_HPP
#define KFUSION_WARP_FIELD_HPP

/**
 * \brief
 * \details
 */
#include <dual_quaternion.hpp>

namespace kfusion
{
    namespace cuda{class TsdfVolume;};
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
     * Translation and rotation of a node, equivalent to dg_se in the paper
     *
     * \var node::weight
     * Equivalent to dg_w
     */
    struct node
    {
        Vec3f vertex;
        kfusion::utils::DualQuaternion<float> transform;
        float weight;
    };
    class WarpField
    {
    public:
        WarpField();
        ~WarpField();

        void init(const cuda::Cloud &frame);
        void init(const std::vector<Vec3f> positions);
        void energy(const cuda::Cloud &frame,
                    const cuda::Normals &normals,
                    const Affine3f &pose,
                    const cuda::TsdfVolume &tsdfVolume,
                    const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
                                                kfusion::utils::DualQuaternion<float>>> &edges
        );

        void energy_data(const cuda::Depth &frame,
                         const Affine3f &pose,
                         const cuda::TsdfVolume &tsdfVolume
        );

        void energy_reg(const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
                                                    kfusion::utils::DualQuaternion<float>>> &edges);
        float tukeyPenaltyFunction(float threshold, float x, float c);
        float huberPenaltyFunction(float threshold, float x, float c);

//        TODO: refactor this to return a 4x4 Matrix - SE(3) form.
        std::vector<node> warp(std::vector<Vec3f> &frame) const;
        utils::DualQuaternion<float> warp(Vec3f point) const;
        utils::DualQuaternion<float> DQB(Vec3f vertex, float voxel_size) const;
        float weighting(Vec3f vertex, Vec3f voxel_center, float weight) const;

//        std::vector<kfusion::utils::DualQuaternion<float>> getQuaternions() const;
        inline void clear(){};

    private:
        std::vector<node> nodes;
    };
}
#endif //KFUSION_WARP_FIELD_HPP
