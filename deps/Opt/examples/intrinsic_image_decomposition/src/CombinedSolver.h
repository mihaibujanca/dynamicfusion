#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>

#include "../../shared/CombinedSolverBase.h"
#include "../../shared/SolverIteration.h"

class CombinedSolver : public CombinedSolverBase
{
	public:
	
        CombinedSolver(const ColorImageR32G32B32A32& image, const CombinedSolverParameters& params) {
			m_image = image;
            m_combinedSolverParameters = params;

            std::vector<unsigned int> dims = { m_image.getWidth(), m_image.getHeight() };
            unsigned int N = dims[0] * dims[1];
            m_targetFloat3 = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
            m_imageFloat3Albedo = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
            m_imageFloatIllumination = createEmptyOptImage(dims, OptImage::Type::FLOAT, 1, OptImage::GPU, true);
		
			resetGPUMemory();
            addOptSolvers(dims, "intrinsic_image_decomposition.t", m_combinedSolverParameters.optDoublePrecision);
		}

        virtual void combinedSolveInit() override { 
            float weightFit = 500.0f;
            float weightRegAlbedo = 1000.0f;
            float weightRegShading = 10000.0f;
            float pNorm = 0.8;
            
            m_weightFitSqrt = sqrtf(weightFit);
            m_weightRegAlbedoSqrt = sqrtf(weightRegAlbedo);
            m_weightRegShadingSqrt = sqrtf(weightRegShading);
            m_pnorm = pNorm;

            m_problemParams.set("w_fitSqrt", &m_weightFitSqrt);
            m_problemParams.set("w_regSqrtAlbedo", &m_weightRegAlbedoSqrt);
            m_problemParams.set("w_regSqrtShading", &m_weightRegShadingSqrt);
            m_problemParams.set("pNorm", &m_pnorm);
            m_problemParams.set("r", m_imageFloat3Albedo);
            m_problemParams.set("i", m_targetFloat3);
            m_problemParams.set("s", m_imageFloatIllumination);

            m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
            m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
        }
        virtual void preSingleSolve() override {
            resetGPUMemory();
        }
        virtual void postSingleSolve() override {
            copyResultToCPUFromFloat3();
        }

        virtual void preNonlinearSolve(int) override {}
        virtual void postNonlinearSolve(int) override {}

        virtual void combinedSolveFinalize() override {
            reportFinalCosts("Intrinsic Image Decomposition", m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"), nan(""));
        }

		void resetGPUMemory()
		{
			std::vector<float3> h_imageFloat3(m_image.getWidth()*m_image.getHeight());
			std::vector<float3> h_imageFloat3Albedo(m_image.getWidth()*m_image.getHeight());
			std::vector<float>  h_imageFloatIllumination(m_image.getWidth()*m_image.getHeight());
			
			for (unsigned int i = 0; i < m_image.getHeight(); i++)
			{
				for (unsigned int j = 0; j < m_image.getWidth(); j++)
				{
					float EPS = 0.01f;

					ml::vec4f v = m_image(j, i);
					v = v / 255.0f;

					float intensity = (v.x + v.y + v.z) / 3.0f;
					ml::vec4f chroma = v / intensity;

					ml::vec4f t = m_image(j, i);
					t = t / 255.0f;

					t.x = log2(t.x + EPS);
					t.y = log2(t.y + EPS);
					t.z = log2(t.z + EPS);
					t.w = 0.0f;
					
					intensity = log2(intensity + EPS);
					
					chroma.x = log2(chroma.x + EPS);
					chroma.y = log2(chroma.y + EPS);
					chroma.z = log2(chroma.z + EPS);
	
					h_imageFloat3[i*m_image.getWidth() + j] = make_float3(t.x, t.y, t.z);
					h_imageFloat3Albedo[i*m_image.getWidth() + j] = make_float3(chroma.x, chroma.y, chroma.z);
					h_imageFloatIllumination[i*m_image.getWidth() + j] = intensity;
				}
			}
            m_targetFloat3->update(h_imageFloat3);
            m_imageFloat3Albedo->update(h_imageFloat3Albedo);
            m_imageFloatIllumination->update(h_imageFloatIllumination);
		}

		ColorImageR32G32B32A32* getAlbedo()
		{
			return &m_result;
		}

		ColorImageR32G32B32A32* getShading()
		{
			return &m_resultShading;
		}

		void copyResultToCPUFromFloat3()
		{
			m_result = ColorImageR32G32B32A32(m_image.getWidth(), m_image.getHeight());

			std::vector<float3> h_imageFloat3(m_image.getWidth()*m_image.getHeight());
            m_imageFloat3Albedo->copyTo(h_imageFloat3);
			for (unsigned int i = 0; i < m_image.getHeight(); i++)
			{
				for (unsigned int j = 0; j < m_image.getWidth(); j++)
				{
					float3 v = h_imageFloat3[i*m_image.getWidth() + j];
					v.x = exp2(v.x) / 1.5f;
					v.y = exp2(v.y) / 1.5f;
					v.z = exp2(v.z) / 1.5f;
					m_result(j, i) = vec4f(v.x, v.y, v.z, 1.0f);
				}
			}

			m_resultShading = ColorImageR32G32B32A32(m_image.getWidth(), m_image.getHeight());
			
            std::vector<float> h_imageFloatShading(m_image.getWidth()*m_image.getHeight());
            m_imageFloatIllumination->copyTo(h_imageFloatShading);

			for (unsigned int i = 0; i < m_image.getHeight(); i++)
			{
				for (unsigned int j = 0; j < m_image.getWidth(); j++)
				{
					float v = h_imageFloatShading[i*m_image.getWidth() + j];
					v = exp2(v);
					m_resultShading(j, i) = vec4f(v, v, v, 1.0f);
				}
			}
		}

	private:
        float m_weightFitSqrt;
        float m_weightRegAlbedoSqrt;
        float m_weightRegShadingSqrt;
        float m_pnorm;

		ColorImageR32G32B32A32 m_result;
		ColorImageR32G32B32A32 m_resultShading;
		ColorImageR32G32B32A32 m_image;
	
		std::shared_ptr<OptImage> m_imageFloat3Albedo;
		std::shared_ptr<OptImage> m_imageFloatIllumination;
		std::shared_ptr<OptImage> m_targetFloat3;
};
