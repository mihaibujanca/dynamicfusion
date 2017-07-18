#ifndef KFUSION_WARP_FIELD_HPP
#define KFUSION_WARP_FIELD_HPP

/**
 * \brief
 * \details
 */
#include <dual_quaternion.hpp>
#include <kfusion/types.hpp>
namespace kfusion
{
    namespace cuda
    {
        class TsdfVolume;
    }
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

        void init(const cuda::Cloud &frame, const cuda::Normals& normals);
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
        inline float tukeyPenalty(float x, float c)
        {
            return std::abs(x) <= c ? x * std::pow((1 - (x * x) / (c * c)), 2) : 0.0;
        }

        /**
         * Huber penalty function, implemented as described in https://en.wikipedia.org/wiki/Huber_loss
         * \param a
         * \param delta
         * \return
         */
        inline float huberPenalty(float a, float delta)
        {
            return std::abs(a) <= delta ? a * a / 2 : delta * std::abs(a) - delta * delta / 2;
        }

        void warp(std::vector<Point, std::allocator<Point>>& cloud_host,
                  std::vector<Point, std::allocator<Point>>& normals_host) const;
        utils::DualQuaternion<float> warp(Vec3f point) const;
        utils::DualQuaternion<float> DQB(Vec3f vertex, float voxel_size) const;
        float weighting(Vec3f vertex, Vec3f voxel_center, float weight) const;

        //        std::vector<kfusion::utils::DualQuaternion<float>> getQuaternions() const;
        inline void clear(){};

    private:
        //    Possibly have an internal kd-tree of nodes rather than a vector?
        std::vector<node> nodes;
    };
}
#endif //KFUSION_WARP_FIELD_HPP