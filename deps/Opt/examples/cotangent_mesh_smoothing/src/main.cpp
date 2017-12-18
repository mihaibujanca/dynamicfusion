#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"

int main(int argc, const char * argv[])
{
    std::string filename = "../data/head.ply";
    if (argc >= 2) {
        filename = argv[1];
    }
    bool performanceRun = false;
    if (argc >= 3) {
        if (std::string(argv[2]) == "perf") {
            performanceRun = true;
        }
        else {
            printf("Invalid second parameter: %s\n", argv[2]);
        }
    }


    SimpleMesh* mesh = new SimpleMesh();
    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename << std::endl;
        exit(1);
    }
    CombinedSolverParameters params;
    params.useOpt = true;
    params.useOptLM = false;
    params.nonLinearIter = 5;
    params.linearIter = 25;
    float weightFit = 1.0f;
    float weightReg = 0.5f;
    CombinedSolver solver(mesh, performanceRun, params, weightFit, weightReg);
    solver.solveAll();
    SimpleMesh* res = solver.result();
    if (!OpenMesh::IO::write_mesh(*res, "out.off"))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << "out.off" << std::endl;
        exit(1);
    }

	return 0;
}
