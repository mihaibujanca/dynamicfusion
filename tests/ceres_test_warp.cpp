#include <cmath>
#include <cstdio>
#include <iostream>
#include <kfusion/warp_field.hpp>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <kfusion/optimisation.hpp>

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
                                 warpProblem.mutable_epsilon());
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
