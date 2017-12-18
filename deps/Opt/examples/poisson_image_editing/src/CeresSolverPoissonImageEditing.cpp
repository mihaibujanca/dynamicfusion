#pragma once

#include "main.h"
#include "../../shared/OptUtils.h"
#include "../../shared/Config.h"
#if USE_CERES
#include <cuda_runtime.h>

#include "CeresSolverPoissonImageEditing.h"
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "ceres/ceres.h"

#include "glog/logging.h"

using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;

vec4f toVec(const float4 &v)
{
    return vec4f(v.x, v.y, v.z, v.w);
}

struct EdgeTerm
{
    EdgeTerm(const vec4f &_targetDelta, float _weight)
        : targetDelta(_targetDelta), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const xA, const T* const xB, T* residuals) const
    {
        for (int i = 0; i < 4; i++)
            residuals[i] = (xA[i] - xB[i] - T(targetDelta[i])) * T(weight);
        /*residuals[0] = xA[0];
        residuals[1] = xA[1];
        residuals[2] = xA[2];
        residuals[3] = xA[3];*/
        return true;
    }

    static ceres::CostFunction* Create(const vec4f &targetDelta, float weight)
    {
        return (new ceres::AutoDiffCostFunction<EdgeTerm, 4, 4, 4>(
            new EdgeTerm(targetDelta, weight)));
    }

    vec4f targetDelta;
    float weight;
};

struct FitTerm
{
    FitTerm(const vec4f &_targetValue, float _weight)
        : targetValue(_targetValue), weight(_weight) {}

    template <typename T>
    bool operator()(const T* const x, T* residuals) const
    {
        for (int i = 0; i < 4; i++)
            residuals[i] = (x[i] - T(targetValue[i])) * T(weight);
        return true;
    }

    static ceres::CostFunction* Create(const vec4f &targetValue, float weight)
    {
        return (new ceres::AutoDiffCostFunction<FitTerm, 4, 4>(
            new FitTerm(targetValue, weight)));
    }

    vec4f targetValue;
    float weight;
};

double CeresSolverPoissonImageEditing::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters)
{

    Problem problem;

    auto getPixel = [=](int x, int y) {
        return y * m_dims[0] + x;
    };

    const int pixelCount = m_dims[0] * m_dims[1];
    std::vector<float4> h_unknownFloat(pixelCount);
    std::vector<float4> h_target(pixelCount);
    std::vector<float> h_mask(pixelCount);

    findAndCopyArrayToCPU("X", h_unknownFloat, problemParameters);
    findAndCopyArrayToCPU("T", h_target, problemParameters);
    findAndCopyArrayToCPU("M", h_mask, problemParameters);

    std::vector<double4> h_unknownDouble(pixelCount);

    for (int i = 0; i < pixelCount; i++)
    {
        h_unknownDouble[i].x = h_unknownFloat[i].x;
        h_unknownDouble[i].y = h_unknownFloat[i].y;
        h_unknownDouble[i].z = h_unknownFloat[i].z;
        h_unknownDouble[i].w = h_unknownFloat[i].w;
    }

    vector< pair<int, int> > edges;
    for (int y = 0; y < m_dims[1] - 1; y++)
    {
        for (int x = 0; x < m_dims[0] - 1; x++)
        {
            int pixel00 = getPixel(x + 0, y + 0);
            int pixel10 = getPixel(x + 1, y + 0);
            int pixel01 = getPixel(x + 0, y + 1);
            int pixel11 = getPixel(x + 1, y + 1);
            edges.push_back(make_pair(pixel00, pixel10));
            edges.push_back(make_pair(pixel00, pixel01));

            edges.push_back(make_pair(pixel11, pixel10));
            edges.push_back(make_pair(pixel11, pixel01));
        }
    }

    cout << "Edges: " << edges.size() << endl;

    int edgesAdded = 0;
    // add all edge constraints
    for (auto &e : edges)
    {
        const float mask = h_mask[e.first];
        if (mask == 0.0f)
        {
            const vec4f targetA = toVec(h_target[e.first]);
            const vec4f targetB = toVec(h_target[e.second]);

            vec4f targetDelta = targetA - targetB;
            ceres::CostFunction* costFunction = EdgeTerm::Create(targetA - targetB, 1.0f);
            double4 *varStartA = h_unknownDouble.data() + e.first;
            double4 *varStartB = h_unknownDouble.data() + e.second;

            problem.AddResidualBlock(costFunction, NULL, (double*)varStartA, (double*)varStartB);
            edgesAdded++;
        }
    }
    cout << "Edges added: " << edgesAdded << endl;

    // add all fit constraints
    set<int> addedEdges;
    for (auto &e : edges)
    {
        const float mask = h_mask[e.first];
        if (mask != 0.0f && addedEdges.count(e.first) == 0)
        {
            addedEdges.insert(e.first);
            const vec4f target = toVec(h_unknownFloat[e.first]);

            ceres::CostFunction* costFunction = FitTerm::Create(target, 1.0f);
            double4 *varStart = h_unknownDouble.data() + e.first;

            problem.AddResidualBlock(costFunction, NULL, (double*)varStart);
            edgesAdded++;
        }
    }

    cout << "Solving..." << endl;

    Solver::Summary summary;
    unique_ptr<Solver::Options> options = initializeOptions(solverParameters);
    options->function_tolerance = 0.01;
    options->gradient_tolerance = 1e-4 * options->function_tolerance;

    double cost = launchProfiledSolveAndSummary(options, &problem, profileSolve, iters);
    m_finalCost = cost;

    for (int i = 0; i < pixelCount; i++)
    {
        h_unknownFloat[i].x = (float)h_unknownDouble[i].x;
        h_unknownFloat[i].y = (float)h_unknownDouble[i].y;
        h_unknownFloat[i].z = (float)h_unknownDouble[i].z;
        h_unknownFloat[i].w = (float)h_unknownDouble[i].w;
    }

    findAndCopyToArrayFromCPU("X", h_unknownFloat, problemParameters);
    return m_finalCost;
}

#endif