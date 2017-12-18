#pragma once
#include <limits>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <string>
#include "CombinedSolverParameters.h"

struct SolverIteration
{
    SolverIteration() {}
    SolverIteration(double _cost, double _timeInMS) { cost = _cost; timeInMS = _timeInMS; }
    double cost = -std::numeric_limits<double>::infinity();
    double timeInMS = -std::numeric_limits<double>::infinity();
};

template<class T>
const T& clampedRead(const std::vector<T> &v, int index)
{
    if (index < 0) return v[0];
    if (index >= v.size()) return v[v.size() - 1];
    return v[index];
}

static void saveSolverResults(std::string directory, std::string suffix,
    const std::vector<SolverIteration>& ceresIters, const std::vector<SolverIteration>& optGNIters, const std::vector<SolverIteration>& optLMIters, bool optDoublePrecision) {
    std::ofstream resultFile(directory + "results" + suffix + ".csv");
    resultFile << std::scientific;
    resultFile << std::setprecision(20);
    std::string colSuffix = optDoublePrecision ? " (double)" : " (float)";
	resultFile << "Iter, Ceres Error, ";
	resultFile << "Opt(GN) Error" << colSuffix << ",  Opt(LM) Error" << colSuffix << ", Ceres Iter Time(ms), ";
	resultFile << "Opt(GN) Iter Time(ms)" << colSuffix << ", Opt(LM) Iter Time(ms)" << colSuffix << ", Total Ceres Time(ms), ";
	resultFile << "Total Opt(GN) Time(ms)" << colSuffix << ", Total Opt(LM) Time(ms)" << colSuffix << std::endl;
    double sumOptGNTime = 0.0;
    double sumOptLMTime = 0.0;
    double sumCeresTime = 0.0;

    auto _ceresIters = ceresIters;
    auto _optLMIters = optLMIters;
    auto _optGNIters = optGNIters;
    
    if (_ceresIters.size() == 0) {
        _ceresIters.push_back(SolverIteration(0, 0));
    }
    if (_optLMIters.size() == 0) {
        _optLMIters.push_back(SolverIteration(0, 0));
    }
    if (_optGNIters.size() == 0) {
        _optGNIters.push_back(SolverIteration(0, 0));
    }
    for (int i = 0; i < (int)std::max((int)_ceresIters.size(), std::max((int)_optLMIters.size(), (int)_optGNIters.size())); i++)
    {
        double ceresTime = ((_ceresIters.size() > i) ? _ceresIters[i].timeInMS : 0.0);
        double optGNTime = ((_optGNIters.size() > i) ? _optGNIters[i].timeInMS : 0.0);
        double optLMTime = ((_optLMIters.size() > i) ? _optLMIters[i].timeInMS : 0.0);
        sumCeresTime += ceresTime;
        sumOptGNTime += optGNTime;
        sumOptLMTime += optLMTime;
        resultFile << i << ", " << clampedRead(_ceresIters, i).cost << ", " << clampedRead(_optGNIters, i).cost << ", " << clampedRead(_optLMIters, i).cost << ", " << ceresTime << ", " << optGNTime << ", " << optLMTime << ", " << sumCeresTime << ", " << sumOptGNTime << ", " << sumOptLMTime << std::endl;
    }
}


static void reportFinalCosts(std::string name, const CombinedSolverParameters& params, double gnCost, double lmCost, double ceresCost) {
    std::cout << "===" << name << "===" << std::endl;
    std::cout << "**Final Costs**" << std::endl;
    std::cout << "Opt GN,Opt LM,CERES" << std::endl;
    std::cout << std::scientific;
    std::cout << std::setprecision(20);
    if (params.useOpt) {
        std::cout << gnCost;
    }
    std::cout << ",";
    if (params.useOptLM) {
        std::cout << lmCost;
    }
    std::cout << ",";
    if (params.useCeres) {
        std::cout << ceresCost;
    }
    std::cout << std::endl;
}
