#include <cmath>
#include <cstdio>
#include <iostream>
#include <kfusion/warp_field.hpp>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
// Read a Bundle Adjustment in the Large dataset.
class WarpProblem {
public:
    WarpProblem(kfusion::WarpField warp) : warpField_(&warp){};
    ~WarpProblem() {
        delete[] epsilon_index;
        delete[] observations_;
        delete[] parameters_;
    }
    int num_observations()       const { return num_observations_;               }
    const double* observations() const { return observations_;                   }
    const cv::Vec3d* observations_vector() const { return observations_vector_;   }
    double* mutable_cameras()          { return parameters_;                     }
    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + epsilon_index[i] * 6;
    }

private:
    template<typename T>
    void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }
    int num_pixels_;
    int num_observations_;
    int num_parameters_;

    int* epsilon_index;

    double* observations_;
    cv::Vec3d* observations_vector_;
    double* parameters_;

    kfusion::WarpField* warpField_;
};

struct DynamicFusionDataEnergy {
    DynamicFusionDataEnergy(cv::Vec3d observed, cv::Vec3f vertex_canonical, kfusion::WarpField* warpField)
            : vertex_live(observed), warpField_(warpField), vertex_canonical_(vertex_canonical) {}
    template <typename T>
    bool operator()(const T* const epsilon,
                    T* residuals) const {
//        auto quaternion = warpField_->DQB(cv::Vec3f((double)vertex[0],(float)vertex[1],(float)vertex[2]));
        T predicted_x, predicted_y, predicted_z;
        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - vertex_live[0];
        residuals[1] = predicted_y - vertex_live[1];
        residuals[2] = predicted_z - vertex_live[2];
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.{
    static ceres::CostFunction* Create(const cv::Vec3d observed,const cv::Vec3f canonical, kfusion::WarpField* warpField) {
        return (new ceres::AutoDiffCostFunction<DynamicFusionDataEnergy, 3, 9>(
                new DynamicFusionDataEnergy(observed, canonical, warpField)));
    }
    const cv::Vec3d vertex_live;
    const cv::Vec3f vertex_canonical_;
    kfusion::WarpField* warpField_;
};


int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    if (argc != 2) {
        std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
        return 1;
    }
    kfusion::WarpField warpField;
    WarpProblem warpProblem(warpField);
    const auto observations_vector = warpProblem.observations_vector();
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    for (int i = 0; i < warpProblem.num_observations(); ++i) {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        cv::Vec3f canonical;
        ceres::CostFunction* cost_function =
                DynamicFusionDataEnergy::Create(observations_vector[i], canonical, &warpField);
        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 warpProblem.mutable_camera_for_observation(i));
    }
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    return 0;
}
