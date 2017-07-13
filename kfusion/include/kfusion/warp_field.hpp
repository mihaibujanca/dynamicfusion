#ifndef KFUSION_WARP_FIELD_HPP
#define KFUSION_WARP_FIELD_HPP

/**
 * \brief a dual quaternion class for encoding transformations.
 * \details transformations are stored as first a translation; then a
 *          rotation. It is possible to switch the order. See this paper:
 *  https://www.thinkmind.org/download.php?articleid=intsys_v6_n12_2013_5
 */
#include <dual_quaternion.hpp>

namespace kfusion
{
//    template<typename T>
    class WarpField
    {
    public:
        WarpField();
        utils::DualQuaternion<float> warp(Vec3f point) const;
        std::vector<kfusion::utils::DualQuaternion<float>> warp(std::vector<Vec3f> &frame) const;
        float weighting(Vec3f vertex, Vec3f voxel_center, float weight) const;
        std::vector<kfusion::utils::DualQuaternion<float>> getQuaternions() const;
        void fetchQuaternions();
        utils::DualQuaternion<float> DQB(Vec3f vertex, float voxel_size) const;

    private:
        template<typename T>
        struct node
        {
            kfusion::utils::DualQuaternion<T> vertex;
            T weight;
        };
        std::vector<node<float>> nodes;
    };
}
#endif //KFUSION_WARP_FIELD_HPP
