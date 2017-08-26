#ifndef KFUSION_OPTIMISATION_H
#define KFUSION_OPTIMISATION_H
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <kfusion/warp_field.hpp>

struct DynamicFusionDataEnergy {
    DynamicFusionDataEnergy(cv::Vec3d observed, cv::Vec3f vertex_canonical, kfusion::WarpField* warpField)
            : vertex_live(observed), warpField_(warpField), vertex_canonical_(vertex_canonical), normal_canonical_(vertex_canonical) {}
    template <typename T>
    bool operator()(const T* const epsilon, T* residuals) const
    {
        float weights[KNN_NEIGHBOURS];
        warpField_->getWeightsAndUpdateKNN(vertex_canonical_, weights);
        auto nodes = warpField_->getNodes();
        T total_quaternion[4];
        T total_translation[3];
        for(int i = 0; i < KNN_NEIGHBOURS; i++)
        {
            auto quat = weights[i] * nodes->at(warpField_->ret_index[i]).transform;
            T t[3];

            T eps_r[3] = {epsilon[6*i],epsilon[6*i + 1],epsilon[6*i + 2]};
            T eps_t[3] = {epsilon[6*i + 3],epsilon[6*i + 4],epsilon[6*i + 5]};
            float temp[3];
            auto r_quat = quat.getRotation();
            double r[4] = {double(r_quat.w_),double(r_quat.x_),double(r_quat.y_),double(r_quat.z_)};
            quat.getTranslation(temp[0], temp[1], temp[2]);


            T eps_quaternion[4];
            ceres::AngleAxisToQuaternion(eps_r, eps_quaternion);
            T product[4];
            //Quaternion product
            product[0] = eps_quaternion[0] * r[0] - eps_quaternion[1] * r[1] - eps_quaternion[2] * r[2] - eps_quaternion[3] * r[3];
            product[1] = eps_quaternion[0] * r[1] + eps_quaternion[1] * r[0] + eps_quaternion[2] * r[3] - eps_quaternion[3] * r[2];
            product[2] = eps_quaternion[0] * r[2] - eps_quaternion[1] * r[3] + eps_quaternion[2] * r[0] + eps_quaternion[3] * r[1];
            product[3] = eps_quaternion[0] * r[3] + eps_quaternion[1] * r[2] - eps_quaternion[2] * r[1] + eps_quaternion[3] * r[0];

            total_quaternion[0] += product[0];
            total_quaternion[1] += product[1];
            total_quaternion[2] += product[2];
            total_quaternion[3] += product[3];

            //probably wrong, should do like in quaternion multiplication.
            total_translation[0] += (double)temp[0] +  eps_t[0];
            total_translation[1] += (double)temp[1] +  eps_t[1];
            total_translation[2] += (double)temp[2] +  eps_t[2];

        }


        T predicted_x, predicted_y, predicted_z;
        T point[3];
        T predicted[3];
        ceres::QuaternionRotatePoint(total_quaternion, point, predicted);
        predicted_x = predicted[0] + total_translation[0];
        predicted_y = predicted[1] + total_translation[1];
        predicted_z = predicted[2] + total_translation[2];

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - vertex_live[0];
        residuals[1] = predicted_y - vertex_live[1];
        residuals[2] = predicted_z - vertex_live[2];
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.{
    static ceres::CostFunction* Create(const cv::Vec3d observed,const cv::Vec3f canonical, kfusion::WarpField* warpField) {
        return (new ceres::AutoDiffCostFunction<DynamicFusionDataEnergy, 1, 6>(
                new DynamicFusionDataEnergy(observed, canonical, warpField)));
    }
    const cv::Vec3d vertex_live;
    const cv::Vec3f vertex_canonical_;
    const cv::Vec3f normal_canonical_;
    const double tukey_delta = 0.01;
    kfusion::WarpField* warpField_;
};

class WarpProblem {
public:
    WarpProblem(kfusion::WarpField warp) : warpField_(&warp)
    {
        parameters_ = new double[warpField_->getNodes()->size() * 6];
    };
    ~WarpProblem() {
       delete[] parameters_;
    }
    int num_observations()                 const  { return num_observations_;               }
    const cv::Vec3d* observations_vector() const  { return observations_vector_;            }
    double* mutable_epsilon()                     { return parameters_;                     }


private:
    int num_pixels_;
    int num_observations_;
    int num_parameters_;

    int* epsilon_index;

    cv::Vec3d* observations_vector_;
    double* parameters_;

    kfusion::WarpField* warpField_;
};

class Optimisation {

};


#endif //KFUSION_OPTIMISATION_H
