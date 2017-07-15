#ifndef KFUSION_WARP_FIELD_HPP
#define KFUSION_WARP_FIELD_HPP

/**
 * \brief
 * \details
 */
#include <dual_quaternion.hpp>

namespace kfusion
{
    struct node
    {
        kfusion::utils::DualQuaternion<float> vertex;
        float weight;
    };
    class WarpField
    {
    public:
        WarpField();
        utils::DualQuaternion<float> warp(Vec3f point) const;
        std::vector<node> warp(std::vector<Vec3f> &frame) const;
        float weighting(Vec3f vertex, Vec3f voxel_center, float weight) const;
        std::vector<kfusion::utils::DualQuaternion<float>> getQuaternions() const;
        utils::DualQuaternion<float> DQB(Vec3f vertex, float voxel_size) const;
        inline void clear(){};

    private:
        std::vector<node> nodes;
    };
}
#endif //KFUSION_WARP_FIELD_HPP
