#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>

#include "ImageHelper.h"
#include "../../shared/CombinedSolverBase.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/SolverIteration.h"


template <typename T>
static void updateOptImage(std::shared_ptr<OptImage> dst, BaseImage<T> src) {
    dst->update((void*)src.getData(), sizeof(T)*src.getWidth()*src.getHeight(), OptImage::Location::CPU);
}

class CombinedSolver : public CombinedSolverBase {
public:
    CombinedSolver(const ColorImageR32& sourceImage, const ColorImageR32& targetImage, const CombinedSolverParameters& params) {
        m_combinedSolverParameters = params;
        const unsigned int numLevels = 2;
        const float sigmas[2] = { 1.0f, 5.0f };

		m_levels.resize(numLevels);
		for (unsigned int i = 0; i < numLevels; i++) {
			ColorImageR32 targetFiltered = targetImage;
			ColorImageR32 sourceFiltered = sourceImage;
			ImageHelper::filterGaussian(targetFiltered, sigmas[i]);
			ImageHelper::filterGaussian(sourceFiltered, sigmas[i]);
			LodePNG::save(ImageHelper::convertGrayToColor(targetFiltered), "target_filtered" + std::to_string(i) + ".png");
			LodePNG::save(ImageHelper::convertGrayToColor(sourceFiltered), "source_filtered" + std::to_string(i) + ".png");
			m_levels[i].init(sourceFiltered, targetFiltered);
		}		
		resetGPU();
        addOptSolvers(m_levels[0].m_dims, "optical_flow.t", m_combinedSolverParameters.optDoublePrecision);
	}


    /** This solver is a bit more complicated than most of the other examples, since it runs in a hierarchical fashion.
        We'll just do a one-off override of the main solve function to handle this. */
    virtual void solveAll() override {
        combinedSolveInit();
        for (auto s : m_solverInfo) {
            if (s.enabled) {
                for (int i = (int)m_levels.size() - 1; i >= 0; i--) {
                    if (i < (int)m_levels.size() - 1) {
                        //init from previous levels if possible
                        m_levels[i].initFlowVectorsFromOther(m_levels[i + 1]);
                    }
                    auto& level = m_levels[i];
                    m_problemParams.set("X", level.m_flowVectors);
                    m_problemParams.set("I", level.m_source);
                    m_problemParams.set("I_hat", level.m_target);
                    m_problemParams.set("I_hat_dx", level.m_targetDU);
                    m_problemParams.set("I_hat_dy", level.m_targetDV);
                    singleSolve(s);
                }
            }
        }
        combinedSolveFinalize();
    }

    virtual void combinedSolveInit() override {
        m_weightFit = 10.0f;

        float weightReg = 0.1f;

        float fitTarget = 50.0f;

        m_fitStepSize = (fitTarget - m_weightFit) / (m_combinedSolverParameters.numIter*m_levels.size());

        m_fitSqrt = sqrtf(m_weightFit);
        m_regSqrt = sqrtf(weightReg);

        m_problemParams.set("w_fit", &m_fitSqrt);
        m_problemParams.set("w_reg", &m_regSqrt);

        m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
        m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
    }
    virtual void preSingleSolve() override {
        resetGPU();
    }
    virtual void postSingleSolve() override { }

    virtual void preNonlinearSolve(int) override {
        m_weightFit += m_fitStepSize;
        m_fitSqrt = sqrtf(m_weightFit);
    }
    virtual void postNonlinearSolve(int) override {}

    virtual void combinedSolveFinalize() override {
        reportFinalCosts("Optical Flow", m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"), nan(""));
    }

	void resetGPU()
	{
		for (size_t i = 0; i < m_levels.size(); i++) {
			m_levels[i].resetFlowVectors();
		}
	}

    BaseImage<float2> result() {
        return m_levels[0].getFlowVectors();
    }

private:

	struct HierarchyLevel {

		void init(const ColorImageR32& source, const ColorImageR32& target) {
			assert(source.getWidth() == target.getWidth() && source.getHeight() == target.getHeight());
            m_dims = { source.getWidth(), source.getHeight() };
            m_source        = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 1, OptImage::Location::GPU, true);
            m_target        = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 1, OptImage::Location::GPU, true);
            m_targetDU      = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 1, OptImage::Location::GPU, true);
            m_targetDV      = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 1, OptImage::Location::GPU, true);
            m_flowVectors   = createEmptyOptImage(m_dims, OptImage::Type::FLOAT, 2, OptImage::Location::GPU, true);

			ColorImageR32 targetDU = computeDU(target);
			ColorImageR32 targetDV = computeDV(target);
            BaseImage<float2> initFlowVectors(m_dims[0], m_dims[1]);
			initFlowVectors.setPixels(make_float2(0.0f, 0.0f));

            updateOptImage(m_source, source);
            updateOptImage(m_target, target);
            updateOptImage(m_targetDU, targetDU);
            updateOptImage(m_targetDV, targetDV);
            updateOptImage(m_flowVectors, initFlowVectors);
            
			LodePNG::save(ImageHelper::convertGrayToColor(targetDU), "testDU.png");
			LodePNG::save(ImageHelper::convertGrayToColor(targetDV), "testDV.png");
		}

		void resetFlowVectors() {
            if (m_dims[0] == 0 || m_dims[1] == 0) return;
            BaseImage<float2> initFlowVectors(m_dims[0], m_dims[1]);
			initFlowVectors.setPixels(make_float2(0.0f, 0.0f));
            m_flowVectors->update((void*)initFlowVectors.getData(), sizeof(float2)*m_dims[0] * m_dims[1], OptImage::Location::CPU);
		}

		void initFlowVectorsFromOther(const HierarchyLevel& level) {
            assert(level.m_dims[0] == m_dims[0] && level.m_dims[1] == m_dims[1]);
            copyImage(m_flowVectors, level.m_flowVectors);
		}

		ColorImageR32 computeDU(const ColorImageR32& image) {
			ColorImageR32 res(image.getWidth(), image.getHeight());
			res.setPixels(0.0f);
			for (unsigned int j = 1; j < image.getHeight() - 1; j++) {
				for (unsigned int i = 1; i < image.getWidth() - 1; i++) {
					float d =
						- image(i - 1, j - 1) - image(i - 1, j) - image(i - 1, j + 1) 
						+ image(i + 1, j - 1) + image(i + 1, j) + image(i + 1, j + 1);
					res(i, j) = d / 8.0f;
				}
			}
			return res;
		}

		ColorImageR32 computeDV(const ColorImageR32& image) {
			ColorImageR32 res(image.getWidth(), image.getHeight());
			res.setPixels(0.0f);
			for (unsigned int j = 1; j < image.getHeight() - 1; j++) {
				for (unsigned int i = 1; i < image.getWidth() - 1; i++) {
					float d =
						- image(i - 1, j - 1) - image(i, j - 1) - image(i + 1, j - 1)
						+ image(i - 1, j + 1) + image(i, j + 1) + image(i + 1, j + 1);
					res(i, j) = d / 8.0f;
				}
			}
			return res;
		}

		BaseImage<float2> getFlowVectors() const {
            BaseImage<float2> flowVectors(m_dims[0], m_dims[1]);
            cudaSafeCall(cudaMemcpy(flowVectors.getData(), m_flowVectors->data(), sizeof(float2)*m_dims[0]*m_dims[1], cudaMemcpyDeviceToHost));
			return flowVectors;
		}

        std::vector<unsigned int>   m_dims;
		std::shared_ptr<OptImage>   m_source;
		std::shared_ptr<OptImage>	m_target;
		std::shared_ptr<OptImage>	m_targetDU;
		std::shared_ptr<OptImage>	m_targetDV;
		std::shared_ptr<OptImage>	m_flowVectors;	//unknowns
	};

    float m_weightFit;
    float m_fitStepSize;

    float m_fitSqrt;
    float m_regSqrt;

    ColorImageR32 m_sourceImage;
    ColorImageR32 m_targetImage;
    ColorImageR32 m_result;
	
	std::vector<HierarchyLevel> m_levels;

};
