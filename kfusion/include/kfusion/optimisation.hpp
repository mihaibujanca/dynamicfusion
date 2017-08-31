#ifndef KFUSION_OPTIMISATION_H
#define KFUSION_OPTIMISATION_H
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <kfusion/warp_field.hpp>

typedef Eigen::Vector3d Vec3;
struct DynamicFusionDataEnergy
{
    DynamicFusionDataEnergy(cv::Vec3d live_vertex,
                            cv::Vec3f live_normal,
                            cv::Vec3f canonical_vertex,
                            cv::Vec3f canonical_normal,
                            kfusion::WarpField *warpField,
                            const float weights[KNN_NEIGHBOURS],
                            const unsigned long ret_index[KNN_NEIGHBOURS])
            : live_vertex_(live_vertex),
              live_normal_(live_normal),
              canonical_vertex_(canonical_vertex),
              canonical_normal_(canonical_normal),
              warpField_(warpField),
              weights_(weights),
              ret_index_(ret_index) {}
    template <typename T>
    bool operator()(T const * const * epsilon_, T* residuals) const
    {
        T const * epsilon = epsilon_[0];
        auto nodes = warpField_->getNodes();

        T total_translation[3] = {T(0), T(0), T(0)};
        float total_translation_float[3] = {0, 0, 0};
        T total_quaternion[4] = {T(0), T(0), T(0), T(0)};

        for(int i = 0; i < KNN_NEIGHBOURS; i++)
        {
            unsigned long ret_index_i = ret_index_[i]; // Index of the node
            auto quat = nodes->at(ret_index_i).transform;
            cv::Vec3f vert;
            quat.getTranslation(vert);

            ret_index_i *= 6;
            T eps_t[3] = {epsilon[ret_index_i + 3],
                          epsilon[ret_index_i + 4],
                          epsilon[ret_index_i + 5]};

            float temp[3];
            quat.getTranslation(temp[0], temp[1], temp[2]);

            total_translation[0] += (T(temp[0]) +  eps_t[0]) * T(weights_[i]);
            total_translation[1] += (T(temp[1]) +  eps_t[1]) * T(weights_[i]);
            total_translation[2] += (T(temp[2]) +  eps_t[2]) * T(weights_[i]);

            total_translation_float[0] += temp[0] * weights_[i];
            total_translation_float[1] += temp[1] * weights_[i];
            total_translation_float[2] += temp[2] * weights_[i];
//            T eps_r[3] = {epsilon[ret_index_i],epsilon[ret_index_i + 1],epsilon[ret_index_i + 2]};
//            T eps_quaternion[4];
//            ceres::AngleAxisToQuaternion(eps_r, eps_quaternion);
//            T product[4];
//            auto r_quat = quat.getRotation();
//            T r[4] = { T(r_quat.w_), T(r_quat.x_), T(r_quat.y_), T(r_quat.z_)};
//
//            ceres::QuaternionProduct(eps_quaternion, r, product);
//
//            total_quaternion[0] += product[0];
//            total_quaternion[1] += product[1];
//            total_quaternion[2] += product[2];
//            total_quaternion[3] += product[3];


        }
//        std::cout<<"FLOATS:"<<total_translation_float[0]<<" "<<total_translation_float[1]<<" "<<total_translation_float[2]<<std::endl;

        residuals[0] = canonical_vertex_[0] - live_vertex_[0] + total_translation[0];
        residuals[1] = canonical_vertex_[1] - live_vertex_[1] + total_translation[1];
        residuals[2] = canonical_vertex_[2] - live_vertex_[2] + total_translation[2];

        return true;
    }

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
 *
 * In the paper, a value of 0.01 is suggested for c
 */
    template <typename T>
    T tukeyPenalty(T x, T c = T(0.01)) const
    {
        //TODO: this seems to mean that 0.01 is the acceptable threshold for x (otherwise return 0 and as such, it converges). Need to check if this is correct
        return ceres::abs(x) <= c ? x * ceres::pow((T(1.0) - (x * x) / (c * c)), 2) : T(0.0);
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
//      TODO: this will only have one residual at the end, remember to change
    static ceres::CostFunction* Create(const cv::Vec3d& live_vertex,
                                       const cv::Vec3d& live_normal,
                                       const cv::Vec3f& canonical_vertex,
                                       const cv::Vec3f& canonical_normal,
                                       kfusion::WarpField* warpField,
                                       const float weights[KNN_NEIGHBOURS],
                                       const unsigned long ret_index[KNN_NEIGHBOURS])
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<DynamicFusionDataEnergy, 4>(
                new DynamicFusionDataEnergy(live_vertex,
                                            live_normal,
                                            canonical_vertex,
                                            canonical_normal,
                                            warpField,
                                            weights,
                                            ret_index));
        cost_function->AddParameterBlock(warpField->getNodes()->size() * 6);
        cost_function->SetNumResiduals(3);
        return cost_function;
    }
    const cv::Vec3d live_vertex_;
    const cv::Vec3d live_normal_;
    const cv::Vec3f canonical_vertex_;
    const cv::Vec3f canonical_normal_;

    const float *weights_;
    const unsigned long *ret_index_;

    kfusion::WarpField *warpField_;
};

class WarpProblem {
public:
    WarpProblem(kfusion::WarpField *warp) : warpField_(warp)
    {
        parameters_ = new double[warpField_->getNodes()->size() * 6];
        mutable_epsilon_ = new double*[KNN_NEIGHBOURS * 6];
    };
    ~WarpProblem() {
        delete[] parameters_;
        for(int i = 0; i < KNN_NEIGHBOURS * 6; i++)
            delete[] mutable_epsilon_[i];
        delete[] mutable_epsilon_;
    }
    double **mutable_epsilon(int *index_list)
    {
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
            for(int j = 0; j < 6; j++)
                mutable_epsilon_[i * 6 + j] = &(parameters_[index_list[i] + j]);
        return mutable_epsilon_;
    }
    double *mutable_params()
    {
        return parameters_;
    }


private:
    double **mutable_epsilon_;
    double *parameters_;

    kfusion::WarpField *warpField_;
};

#endif //KFUSION_OPTIMISATION_H
