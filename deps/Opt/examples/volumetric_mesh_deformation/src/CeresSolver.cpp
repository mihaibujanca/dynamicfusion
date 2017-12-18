#include "CeresSolver.h"
#include "../../shared/Config.h"
#if USE_CERES

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include "../../shared/SolverIteration.h"
#include "../../shared/OptUtils.h"
using ceres::Solve;
using namespace std;

/*
local x = Offset(0,0,0)

--fitting
local constraint = Constraints(0,0,0)	-- float3

local e_fit = x - constraint
e_fit = ad.select(ad.greatereq(constraint(0), -999999.9), e_fit, ad.Vector(0.0, 0.0, 0.0))
terms:insert(w_fitSqrt*e_fit)
*/

struct FitTerm
{
	FitTerm(const vec3f &_constraint, float _weight)
		: constraint(_constraint), weight(_weight) {}

	template <typename T>
	bool operator()(const T* const x, T* residuals) const
	{
		residuals[0] = (x[0] - T(constraint.x)) * T(weight);
		residuals[1] = (x[1] - T(constraint.y)) * T(weight);
		residuals[2] = (x[2] - T(constraint.z)) * T(weight);
		return true;
	}

	static ceres::CostFunction* Create(const vec3f &constraint, float weight)
	{
		return (new ceres::AutoDiffCostFunction<FitTerm, 3, 3>(
			new FitTerm(constraint, weight)));
	}

	vec3f constraint;
	float weight;
};

template<class T>
struct vec3T
{
	vec3T() {}
	vec3T(T _x, T _y, T _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}
	T sqMagintude()
	{
		return x * x + y * y + z * z;
	}
	vec3T operator * (T v)
	{
		return vec3T(v * x, v * y, v * z);
	}
	vec3T operator + (const vec3T &v)
	{
		return vec3T(x + v.x, y + v.y, z + v.z);
	}
	const T& operator [](int k) const
	{
		if (k == 0) return x;
		if (k == 1) return y;
		if (k == 2) return z;
		return x;
	}
	T x, y, z;
};

template<class T>
void evalRot(T CosAlpha, T CosBeta, T CosGamma, T SinAlpha, T SinBeta, T SinGamma, T R[9])
{
	R[0] = CosGamma*CosBeta;
	R[1] = -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha;
	R[2] = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;
	R[3] = SinGamma*CosBeta;
	R[4] = CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha;
	R[5] = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;
	R[6] = -SinBeta;
	R[7] = CosBeta*SinAlpha;
	R[8] = CosBeta*CosAlpha;
}

template<class T>
void evalR(T alpha, T beta, T gamma, T R[9])
{
	evalRot(cos(alpha), cos(beta), cos(gamma), sin(alpha), sin(beta), sin(gamma), R);
}

template<class T>
vec3T<T> mul(T matrix[9], const vec3T<T> &v)
{
	vec3T<T> result;
	result.x = matrix[0] * v[0] + matrix[1] * v[1] + matrix[2] * v[2];
	result.y = matrix[3] * v[0] + matrix[4] * v[1] + matrix[5] * v[2];
	result.z = matrix[6] * v[0] + matrix[7] * v[1] + matrix[8] * v[2];
	return result;
}

/*
--regularization
local a = Angle(0,0,0)				-- rotation : float3
local R = evalR(a(0), a(1), a(2))	-- rotation : float3x3
local xHat = UrShape(0,0,0)			-- uv-urshape : float3

local offsets = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}}
for iii ,o in ipairs(offsets) do
	local i,j,k = unpack(o)
	local n = Offset(i,j,k)

	local ARAPCost = (x - n) - mul(R, (xHat - UrShape(i,j,k)))
	local ARAPCostF = ad.select(opt.InBounds(0,0,0),	ad.select(opt.InBounds(i,j,k), ARAPCost, ad.Vector(0.0, 0.0, 0.0)), ad.Vector(0.0, 0.0, 0.0))
	terms:insert(w_regSqrt*ARAPCostF)
end
*/

struct RegTerm
{
	RegTerm(const vec3f &_deltaUr, float _weight)
		: deltaUr(_deltaUr), weight(_weight) {}

	template <typename T>
	bool operator()(const T* const xA, const T* const xB, const T* const a, T* residuals) const
	{
		//cost = (x(i, j) - x(i + ox, j + oy)) - mul(R, urshape(i, j) - urshape(i + ox, j + oy)) * w_regSqrt
		T R[9];
		evalR(a[0], a[1], a[2], R);

		vec3T<T> urOut = mul(R, vec3T<T>(T(deltaUr.x), T(deltaUr.y), T(deltaUr.z)));

		T deltaX[3];
		deltaX[0] = xA[0] - xB[0];
		deltaX[1] = xA[1] - xB[1];
		deltaX[2] = xA[2] - xB[2];

		residuals[0] = (deltaX[0] - urOut[0]) * T(weight);
		residuals[1] = (deltaX[1] - urOut[1]) * T(weight);
		residuals[2] = (deltaX[2] - urOut[2]) * T(weight);
		return true;
	}

	static ceres::CostFunction* Create(const vec3f &deltaUr, float weight)
	{
		return (new ceres::AutoDiffCostFunction<RegTerm, 3, 3, 3, 3>(
			new RegTerm(deltaUr, weight)));
	}

	vec3f deltaUr;
	float weight;
};

vec3f toVec(const float3 &v)
{
	return vec3f(v.x, v.y, v.z);
}

double CeresSolver::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters)
{
    size_t voxelCount = m_dims[0] * m_dims[1] * m_dims[2];
    std::vector<vec3d> vertexPosDouble3(voxelCount);
    std::vector<vec3d> anglesDouble3(voxelCount);

    std::vector<float3> h_vertexPosFloat3(voxelCount);
    std::vector<float3> h_anglesFloat3(voxelCount);
    std::vector<float3> h_vertexPosFloat3Urshape(voxelCount);
    std::vector<float3> h_vertexPosTargetFloat3(voxelCount);

    findAndCopyArrayToCPU("Offset", h_vertexPosFloat3, problemParameters);
    findAndCopyArrayToCPU("Angle", h_anglesFloat3, problemParameters);
    findAndCopyArrayToCPU("UrShape", h_vertexPosFloat3Urshape, problemParameters);
    findAndCopyArrayToCPU("Constraints", h_vertexPosTargetFloat3, problemParameters);

    float weightFitSqrt = getTypedParameter<float>("w_fitSqrt", problemParameters);
    float weightRegSqrt = getTypedParameter<float>("w_regSqrt", problemParameters);

	auto getVoxel = [=](int x, int y, int z) {
        return z * m_dims[0] * m_dims[1] + y * m_dims[0] + x;
	};
	auto voxelValid = [=](int x, int y, int z) {
        return (x >= 0 && x < (int)m_dims[0]) &&
            (y >= 0 && y < (int)m_dims[1]) &&
            (z >= 0 && z < (int)m_dims[2]);
	};

	for (int i = 0; i < (int)voxelCount; i++)
	{
		vertexPosDouble3[i].x = h_vertexPosFloat3[i].x;
		vertexPosDouble3[i].y = h_vertexPosFloat3[i].y;
		vertexPosDouble3[i].z = h_vertexPosFloat3[i].z;
		anglesDouble3[i].x = h_anglesFloat3[i].x;
		anglesDouble3[i].y = h_anglesFloat3[i].y;
		anglesDouble3[i].z = h_anglesFloat3[i].z;
	}

	Problem problem;

	// add all fit constraints
	//if (mask(i, j) == 0 && constaints(i, j).u >= 0 && constaints(i, j).v >= 0)
	//    fit = (x(i, j) - constraints(i, j)) * w_fitSqrt
	for (int i = 0; i < (int)voxelCount; i++)
	{
		const vec3f constraint = toVec(h_vertexPosTargetFloat3[i]);
		if (constraint.x > -999999.9f)
		{
			ceres::CostFunction* costFunction = FitTerm::Create(constraint, weightFitSqrt);
			vec3d *varStart = vertexPosDouble3.data() + i;
			problem.AddResidualBlock(costFunction, NULL, (double *)varStart);
		}
	}

	//local offsets = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } }
	vector<vec3i> offsets;
	offsets.push_back(vec3i(1, 0, 0));
	offsets.push_back(vec3i(-1, 0, 0));
	offsets.push_back(vec3i(0, 1, 0));
	offsets.push_back(vec3i(0, -1, 0));
	offsets.push_back(vec3i(0, 0, 1));
	offsets.push_back(vec3i(0, 0, -1));

    for (int x = 0; x < (int)m_dims[0]; x++)
        for (int y = 0; y < (int)m_dims[1]; y++)
            for (int z = 0; z < (int)m_dims[2]; z++)
			{
				for (vec3i o : offsets)
				{
					const int myIndex = getVoxel(x, y, z);
					const int neighborIndex = getVoxel(x + o.x, y + o.y, z + o.z);
					if (!voxelValid(x + o.x, y + o.y, z + o.z)) continue;

					//const vec3f constraintA = toVec(h_vertexPosTargetFloat3[myIndex]);
					//const vec3f constraintB = toVec(h_vertexPosTargetFloat3[neighborIndex]);
					//if (constraintA.x > -999999.0f && constraintB.x > -999999.0f)
					{
						const vec3f deltaUr = toVec(h_vertexPosFloat3Urshape[myIndex]) - toVec(h_vertexPosFloat3Urshape[neighborIndex]);
						ceres::CostFunction* costFunction = RegTerm::Create(deltaUr, weightRegSqrt);
						vec3d *varStartA = vertexPosDouble3.data() + myIndex;
                        vec3d *varStartB = vertexPosDouble3.data() + neighborIndex;
                        vec3d *angleStartA = anglesDouble3.data()  + myIndex;
						problem.AddResidualBlock(costFunction, NULL, (double*)varStartA, (double*)varStartB, (double*)angleStartA);
					}
				}
			}

	cout << "Solving..." << endl;

    unique_ptr<Solver::Options> options = initializeOptions(solverParameters);

    double cost = launchProfiledSolveAndSummary(options, &problem, profileSolve, iters);
    m_finalCost = cost;

	for (int i = 0; i < (int)voxelCount; i++)
	{
		h_vertexPosFloat3[i].x = (float)vertexPosDouble3[i].x;
		h_vertexPosFloat3[i].y = (float)vertexPosDouble3[i].y;
		h_vertexPosFloat3[i].z = (float)vertexPosDouble3[i].z;
		h_anglesFloat3[i].x = (float)anglesDouble3[i].x;
		h_anglesFloat3[i].y = (float)anglesDouble3[i].y;
		h_anglesFloat3[i].z = (float)anglesDouble3[i].z;
	}
    findAndCopyToArrayFromCPU("Offset", h_vertexPosFloat3, problemParameters);
    findAndCopyToArrayFromCPU("Angle", h_anglesFloat3, problemParameters);
    return m_finalCost;
}

#endif // USE_CERES
