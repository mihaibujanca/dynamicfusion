#include "mLibInclude.h"
#include "../../shared/OptUtils.h"
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <Eigen33b1/Eigen>
#include <Eigen33b1/IterativeLinearSolvers>
#include <cuda_runtime.h>

#include <time.h>

#define LEAST_SQ_CONJ_GRADIENT 0
#define SPARSE_QR 1

#define SOLVER LEAST_SQ_CONJ_GRADIENT

typedef Eigen::Triplet<float> Tripf;
typedef Eigen::SparseMatrix<float> SpMatrixf;
#include "EigenSolverPoissonImageEditing.h"
#include <Eigen33b1/OrderingMethods>

#if SOLVER == LEAST_SQ_CONJ_GRADIENT
typedef Eigen::LeastSquaresConjugateGradient<SpMatrixf > AxEqBSolver;
#elif SOLVER == SPARSE_QR
typedef Eigen::SparseQR<SpMatrixf, Eigen::COLAMDOrdering<int> > AxEqBSolver;
#endif

struct vec2iHash {
    size_t operator()(const vec2i& v) const {
        return std::hash < int > {}(v.x) ^ std::hash < int > {}(v.y);
    }
};


float4 sampleImage(float4* image, vec2i p, int W) {
    return image[p.y*W + p.x];
}

void setPixel(float4* image, vec2i p, int W, float r, float g, float b) {
    image[p.y*W + p.x].x = r;
    image[p.y*W + p.x].y = g;
    image[p.y*W + p.x].z = b;
}

void solveAxEqb(AxEqBSolver& solver, const Eigen::VectorXf& b, Eigen::VectorXf& x) {
    
#if SOLVER==LEAST_SQ_CONJ_GRADIENT
    x = solver.solveWithGuess(b, x);
    //std::cout << "#iterations:     " << solver.iterations() << std::endl;
    //std::cout << "estimated error: " << solver.error() << std::endl;
#else
    x = solver.solve(b);
#endif
}
double EigenSolverPoissonImageEditing::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters)
{
    int numUnknowns = 0;
    std::unordered_map<vec2i, int, vec2iHash> pixelLocationsToIndex;
    std::vector<vec2i> pixelLocations;
    size_t pixelCount = m_dims[0] * m_dims[1];
    std::vector<float4> h_unknownFloat(pixelCount);
    std::vector<float4> h_target(pixelCount);
    std::vector<float>  h_mask(pixelCount);

    findAndCopyArrayToCPU("X", h_unknownFloat, problemParameters);
    findAndCopyArrayToCPU("T", h_target, problemParameters);
    findAndCopyArrayToCPU("M", h_mask, problemParameters);

    for (int y = 0; y < (int)m_dims[1]; ++y) {
        for (int x = 0; x < (int)m_dims[0]; ++x) {
            if (h_mask[y*m_dims[0] + x] == 0.0f) {
                ++numUnknowns;
                vec2i p(x, y);
                pixelLocationsToIndex[p] =(int)pixelLocations.size();
                pixelLocations.push_back(p);
            }
        }
    }
    printf("# Unknowns: %d\n", numUnknowns);
    int numResiduals = (int)pixelLocations.size() * 4;

    Eigen::VectorXf x_r(numUnknowns), b_r(numResiduals);
    Eigen::VectorXf x_g(numUnknowns), b_g(numResiduals);
    Eigen::VectorXf x_b(numUnknowns), b_b(numResiduals);
    Eigen::VectorXf x_a(numUnknowns), b_a(numResiduals);

    b_r.setZero();
    b_g.setZero();
    b_b.setZero();
    b_a.setZero();

    for (int i = 0; i < pixelLocations.size(); ++i) {
        vec2i p = pixelLocations[i];
        float4 color = sampleImage(h_unknownFloat.data(), p, m_dims[0]);
        x_r[i] = color.x;
        //printf("%f\n", color.x);
        x_g[i] = color.y;
        x_b[i] = color.z;
        x_a[i] = color.w;
    }
    SpMatrixf A(numResiduals, numUnknowns);
    A.setZero();
    printf("Constructing Matrix\n");
    std::vector<Tripf> entriesA;

    std::vector<vec2i> offsets;
    offsets.push_back(vec2i(-1, 0));
    offsets.push_back(vec2i(1, 0));
    offsets.push_back(vec2i(0, -1));
    offsets.push_back(vec2i(0, 1));

    for (int i = 0; i < pixelLocations.size(); ++i) {
        vec2i p = pixelLocations[i];
        int numInternalNeighbors = 0;
        float4 g_p = sampleImage(h_target.data(), p, m_dims[0]);
        int j = 0;

        for (vec2i off : offsets) {
            vec2i q = p + off;
            if (q.x >= 0 && q.y >= 0 && q.x < (int)m_dims[0] && q.y < (int)m_dims[1]) {
                auto it = pixelLocationsToIndex.find(q);
                int row = 4 * i + j;
                if (it == pixelLocationsToIndex.end()) {
                    float4 f_q = sampleImage(h_unknownFloat.data(), q, m_dims[0]);
                    b_r[row] += f_q.x;
                    b_g[row] += f_q.y;
                    b_b[row] += f_q.z;
                    b_a[row] += f_q.w;
                }
                else {
                    entriesA.push_back(Tripf(row, it->second, -1.0f));
                }
                entriesA.push_back(Tripf(row, i, 1.0f));

                float4 g_q = sampleImage(h_target.data(), q, m_dims[0]);
                b_r[row] += (g_p.x - g_q.x);
                b_g[row] += (g_p.y - g_q.y);
                b_b[row] += (g_p.z - g_q.z);
                b_a[row] += (g_p.w - g_q.w);
            }
            ++j;
            
        }
    }
    

    printf("Entries Set\n");
    A.setFromTriplets(entriesA.begin(), entriesA.end());
    printf("Sparse Matrix Constructed\n");
    A.makeCompressed();
    printf("Matrix Compressed\n");
    {
        float totalCost = 0.0f;
        
        float cost_r = (A*x_r - b_r).squaredNorm();
        float cost_g = (A*x_g - b_g).squaredNorm();
        float cost_b = (A*x_b - b_b).squaredNorm();
        float cost_a = (A*x_a - b_a).squaredNorm();
        totalCost = cost_r + cost_g + cost_b + cost_a;
        printf("Initial Cost: %f : (%f, %f, %f, %f)\n", totalCost, cost_r, cost_g, cost_b, cost_a);

    }
    

    AxEqBSolver solver;
    solver.setMaxIterations(97);
    printf("Solvers Initialized\n");

    clock_t start = clock(), diff;
    
    solver.compute(A);
    //printf("solver.compute(A)\n");
    solveAxEqb(solver, b_r, x_r);
    //printf("Red solve done\n");
    solveAxEqb(solver, b_g, x_g);
    //printf("Green solve done\n");
    solveAxEqb(solver, b_b, x_b);
    //printf("Blue solve done\n");
    solveAxEqb(solver, b_a, x_a);

    diff = clock() - start;
    printf("Time taken %f ms\n", diff*1000.0 / double(CLOCKS_PER_SEC));

    float totalCost = 0.0f;
 
    float cost_r = (A*x_r - b_r).squaredNorm(); 
    float cost_g = (A*x_g - b_g).squaredNorm();
    float cost_b = (A*x_b - b_b).squaredNorm();
    float cost_a = (A*x_a - b_a).squaredNorm();
    totalCost = cost_r + cost_g + cost_b + cost_a;
    printf("Final Cost: %f : (%f, %f, %f, %f)\n", totalCost, cost_r, cost_g, cost_b, cost_a);

    for (int i = 0; i < pixelLocations.size(); ++i) {
        setPixel(h_unknownFloat.data(), pixelLocations[i], m_dims[0], x_r[i], x_g[i], x_b[i]);
    }
    findAndCopyToArrayFromCPU("X", h_unknownFloat, problemParameters);;
    return (double)totalCost;

}


/* Proper Poisson Image Editing

for (int i = 0; i < pixelLocations.size(); ++i) {
vec2i p = pixelLocations[i];
int row = i;
int numInternalNeighbors = 0;
float4 g_p = sampleImage(h_target, p, m_width);
for (int off_y = -1; off_y <= 1; off_y += 2) {
for (int off_x = -1; off_x <= 1; off_x += 2) {
vec2i q(p.x + off_x, p.y + off_y);
auto it = pixelLocationsToIndex.find(q);
if (it != pixelLocationsToIndex.end()) {
++numInternalNeighbors;
entriesA.push_back(Tripf(row, it->second, -1.0f));
} else {

float4 f_star_q = sampleImage(h_target, q, m_width);
b_r[i] += f_star_q.x;
b_g[i] += f_star_q.y;
b_b[i] += f_star_q.z;
}
float4 g_q = sampleImage(h_target, q, m_width);
b_r[i] += (g_p.x - g_q.x);
b_g[i] += (g_p.y - g_q.y);
b_b[i] += (g_p.z - g_q.z);
}
}
entriesA.push_back(Tripf(row, row, (float)numInternalNeighbors));
}

*/
