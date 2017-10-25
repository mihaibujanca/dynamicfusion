#include "kfusion/warp_field_optimiser.hpp"
kfusion::WarpFieldOptimiser::WarpFieldOptimiser(WarpField *warp, CombinedSolver *solver) : solver_(solver), warp_(warp){}
kfusion::WarpFieldOptimiser::WarpFieldOptimiser(WarpField *warp, CombinedSolverParameters params) : warp_(warp)
{
    solver_ = new CombinedSolver(warp, params);
}
void kfusion::WarpFieldOptimiser::optimiseWarpData(const std::vector<Vec3f> &canonical_vertices,
                                                   const std::vector<Vec3f> &canonical_normals,
                                                   const std::vector<Vec3f> &live_vertices,
                                                   const std::vector<Vec3f> &live_normals)
{
    solver_->initializeProblemInstance(canonical_vertices,
                                       canonical_normals,
                                       live_vertices,
                                       live_normals);
    solver_->solveAll();
}