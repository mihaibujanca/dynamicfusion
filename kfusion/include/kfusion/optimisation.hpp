#ifndef KFUSION_OPTIMISATION_H
#define KFUSION_OPTIMISATION_H
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <kfusion/warp_field.hpp>

struct DynamicFusionDataEnergy
{
    DynamicFusionDataEnergy(const cv::Vec3f& live_vertex,
                            const cv::Vec3f& live_normal,
                            const cv::Vec3f& canonical_vertex,
                            const cv::Vec3f& canonical_normal,
                            kfusion::WarpField *warpField,
                            const float weights[KNN_NEIGHBOURS],
                            const unsigned long knn_indices[KNN_NEIGHBOURS])
            : live_vertex_(live_vertex),
              live_normal_(live_normal),
              canonical_vertex_(canonical_vertex),
              canonical_normal_(canonical_normal),
              warpField_(warpField)
    {
        weights_ = new float[KNN_NEIGHBOURS];
        knn_indices_ = new unsigned long[KNN_NEIGHBOURS];
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
        {
            weights_[i] = weights[i];
            knn_indices_[i] = knn_indices[i];
        }
    }
    ~DynamicFusionDataEnergy()
    {
        delete[] weights_;
        delete[] knn_indices_;
    }
    template <typename T>
    bool operator()(T const * const * epsilon_, T* residuals) const
    {
        auto nodes = warpField_->getNodes();

        T total_translation[3] = {T(0), T(0), T(0)};
        float total_translation_float[3] = {0, 0, 0};

        for(int i = 0; i < KNN_NEIGHBOURS; i++)
        {
            auto quat = nodes->at(knn_indices_[i]).transform;
            cv::Vec3f vert;
            quat.getTranslation(vert);

            T eps_t[3] = {epsilon_[i][3], epsilon_[i][4], epsilon_[i][5]};

            float temp[3];
            quat.getTranslation(temp[0], temp[1], temp[2]);

//            total_translation[0] += (T(temp[0]) +  eps_t[0]);
//            total_translation[1] += (T(temp[1]) +  eps_t[1]);
//            total_translation[2] += (T(temp[2]) +  eps_t[2]);
//
            total_translation[0] += (T(temp[0]) +  eps_t[0]) * T(weights_[i]);
            total_translation[1] += (T(temp[1]) +  eps_t[1]) * T(weights_[i]);
            total_translation[2] += (T(temp[2]) +  eps_t[2]) * T(weights_[i]);

        }

        residuals[0] = T(live_vertex_[0] - canonical_vertex_[0]) - total_translation[0];
        residuals[1] = T(live_vertex_[1] - canonical_vertex_[1]) - total_translation[1];
        residuals[2] = T(live_vertex_[2] - canonical_vertex_[2]) - total_translation[2];

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
        return ceres::abs(x) <= c ? x * ceres::pow((T(1.0) - (x * x) / (c * c)), 2) : T(0.0);
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
//      TODO: this will only have one residual at the end, remember to change
    static ceres::CostFunction* Create(const cv::Vec3f& live_vertex,
                                       const cv::Vec3f& live_normal,
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
        for(int i=0; i < KNN_NEIGHBOURS; i++)
            cost_function->AddParameterBlock(6);
        cost_function->SetNumResiduals(3);
        return cost_function;
    }
    const cv::Vec3f live_vertex_;
    const cv::Vec3f live_normal_;
    const cv::Vec3f canonical_vertex_;
    const cv::Vec3f canonical_normal_;

    float *weights_;
    unsigned long *knn_indices_;

    kfusion::WarpField *warpField_;
};

struct DynamicFusionRegEnergy
{
    DynamicFusionRegEnergy(){};
    ~DynamicFusionRegEnergy(){};
    template <typename T>
    bool operator()(T const * const * epsilon_, T* residuals) const
    {
        return true;
    }

/**
 * Huber penalty function, implemented as described in https://en.wikipedia.org/wiki/Huber_loss
 * In the paper, a value of 0.0001 is suggested for delta.
 * \param a
 * \param delta
 * \return
 */
    template <typename T>
    T huberPenalty(T a, T delta = 0.0001) const
    {
        return ceres::abs(a) <= delta ? a * a / 2 : delta * ceres::abs(a) - delta * delta / 2;
    }

    static ceres::CostFunction* Create()
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<DynamicFusionRegEnergy, 4>(
                new DynamicFusionRegEnergy());
        for(int i=0; i < KNN_NEIGHBOURS; i++)
            cost_function->AddParameterBlock(6);
        cost_function->SetNumResiduals(3);
        return cost_function;
    }
};

class WarpProblem {
public:
    explicit WarpProblem(kfusion::WarpField *warp) : warpField_(warp)
    {
        parameters_ = new double[warpField_->getNodes()->size() * 6];
        for(int i = 0; i < warp->getNodes()->size() * 6; i+=6)
        {
            auto transform = warp->getNodes()->at(i / 6).transform;

            float x,y,z;

            transform.getTranslation(x,y,z);
            parameters_[i] = x;
            parameters_[i+1] = y;
            parameters_[i+2] = z;

            transform.getRotation().getRodrigues(x,y,z);
            parameters_[i+3] = x;
            parameters_[i+4] = y;
            parameters_[i+5] = z;
        }
    };

    ~WarpProblem() {
        delete parameters_;
    }
    std::vector<double*> mutable_epsilon(const unsigned long *index_list) const
    {
        std::vector<double*> mutable_epsilon_(KNN_NEIGHBOURS);
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
            mutable_epsilon_[i] = &(parameters_[index_list[i] * 6]); // Blocks of 6
        return mutable_epsilon_;
    }

    std::vector<double*> mutable_epsilon(const std::vector<size_t>& index_list) const
    {
        std::vector<double*> mutable_epsilon_(KNN_NEIGHBOURS);
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
            mutable_epsilon_[i] = &(parameters_[index_list[i] * 6]); // Blocks of 6
        return mutable_epsilon_;
    }
    double *mutable_params()
    {
        return parameters_;
    }

    const double *params() const
    {
        return parameters_;
    }


    void updateWarp()
    {
        for(int i = 0; i < warpField_->getNodes()->size() * 6; i+=6)
        {
            warpField_->getNodes()->at(i / 6).transform.encodeRotation(parameters_[i],parameters_[i+1],parameters_[i+2]);
            warpField_->getNodes()->at(i / 6).transform.encodeTranslation(parameters_[i+3],parameters_[i+4],parameters_[i+5]);
        }
    }


private:
    double *parameters_;
    kfusion::WarpField *warpField_;
};

#endif //KFUSION_OPTIMISATION_H
