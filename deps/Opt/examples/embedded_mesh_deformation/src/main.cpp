#include "mLibInclude.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"
#include "LandMarkSet.h"

int main(int argc, const char * argv[])
{
	std::string filename = "../data/raptor_simplify2k.off";
	if (argc >= 2) {
		filename = argv[1];
	}

    // For now, any model must be accompanied with a identically 
    // named (besides the extension, which must be 3 characters) mrk file
    std::string markerFilename = filename.substr(0, filename.size() - 3) + "mrk";
	// Load Constraints
	LandMarkSet markersMesh;
    markersMesh.loadFromFile(markerFilename.c_str());

	std::vector<int>				constraintsIdx;
	std::vector<std::vector<float>> constraintsTarget;

	for (unsigned int i = 0; i < markersMesh.size(); i++)
	{
        printf("%d: %d\n", i, (markersMesh[i].getVertexIndex()));
		constraintsIdx.push_back(markersMesh[i].getVertexIndex());
		constraintsTarget.push_back(markersMesh[i].getPosition());
	}

	SimpleMesh* mesh = new SimpleMesh();
	if (!OpenMesh::IO::read_mesh(*mesh, filename))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "bunny.off" << std::endl;
		exit(1);
	}
	printf("Beginning MeshDeformationED Warp\n");

    CombinedSolverParameters params;
    /*params.useOpt = true;
    params.useOptLM = false;
    params.numIter = 32;
    params.nonLinearIter = 1;
    params.linearIter = 4000;
    params.earlyOut = false;
    */

    /* LM is good here */
    params.useOpt = false;
    params.useOptLM = true;
    params.numIter = 31;
    params.nonLinearIter = 5;
    params.linearIter = 125;

    CombinedSolver solver(mesh, constraintsIdx, constraintsTarget, params);
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
