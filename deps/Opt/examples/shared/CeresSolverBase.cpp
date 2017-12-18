#pragma once
#include "CeresSolverBase.h"
#include <core-util/timer.h>
using namespace std;
#if USE_CERES
std::unique_ptr<ceres::Solver::Options> CeresSolverBase::initializeOptions(const NamedParameters& solverParameters) const {
    std::unique_ptr<ceres::Solver::Options> options = std::unique_ptr<ceres::Solver::Options>(new ceres::Solver::Options());
    options->num_threads = 8;
    options->num_linear_solver_threads = 8;
    options->linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
    options->max_num_iterations = 10000;
    options->function_tolerance = 1e-3;
    options->gradient_tolerance = 1e-4 * options->function_tolerance;
    return options;
}

double CeresSolverBase::launchProfiledSolveAndSummary(const std::unique_ptr<ceres::Solver::Options>& options, ceres::Problem* problem, bool profileSolve, std::vector<SolverIteration>& iters) {
    Solver::Summary summary;
    double elapsedTime;
    {
        ml::Timer timer;
        Solve(*options, problem, &summary);
        elapsedTime = timer.getElapsedTimeMS();
    }

    cout << "Solver used: " << summary.linear_solver_type_used << endl;
    cout << "Minimizer iters: " << summary.iterations.size() << endl;
    cout << "Total time: " << elapsedTime << "ms" << endl;

    double iterationTotalTime = 0.0;
    int totalLinearItereations = 0;
    for (auto &i : summary.iterations)
    {
        iterationTotalTime += i.iteration_time_in_seconds;
        totalLinearItereations += i.linear_solver_iterations;
        cout << "Iteration: " << i.linear_solver_iterations << " " << i.iteration_time_in_seconds * 1000.0 << "ms," << " cost: " << i.cost << endl;
    }
    if (profileSolve) {
        for (auto &i : summary.iterations) {
            iters.push_back(SolverIteration(i.cost, i.iteration_time_in_seconds * 1000.0));
        }
    }


    cout << "Total iteration time: " << iterationTotalTime << endl;
    cout << "Cost per linear solver iteration: " << iterationTotalTime * 1000.0 / totalLinearItereations << "ms" << endl;

    double cost = -1.0;
    problem->Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    cout << "Cost end: " << cost << endl;
    cout << summary.FullReport() << endl;
    return cost;
}
#endif