#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"

static SimpleMesh* createMesh(std::string filename) {
    SimpleMesh* mesh = new SimpleMesh();
    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename << std::endl;
        exit(1);
    }
    printf("Faces: %d\nVertices: %d\n", (int)mesh->n_faces(), (int)mesh->n_vertices());
    return mesh;
}

static std::vector<int4> getSourceTetIndices(std::string filename) {
    // TODO: error handling
    std::ifstream inFile(filename);
    int tetCount = 0;
    int temp;
    inFile >> tetCount >> temp >> temp;
    std::vector<int4> tets(tetCount);
    for (int i = 0; i < tetCount; ++i) {
        inFile >> temp >> tets[i].x >> tets[i].y >> tets[i].z >> tets[i].w;
    }
    int4 f = tets[tets.size() - 1];
    printf("Final tet read: %d %d %d %d\n", f.x, f.y, f.z, f.w);
    return tets;
}

int main(int argc, const char * argv[])
{
    std::string targetSourceDirectory = "../data/squat_target";
    std::string sourceFilename = "../data/squat_source.obj";
    std::string tetmeshFilename = "../data/squat_tetmesh.ele";

    if (argc > 1) {
        assert(argc > 3);
        targetSourceDirectory = argv[1];
        sourceFilename = argv[2];
        tetmeshFilename = argv[3];
    }

    std::vector<std::string> targetFiles = ml::Directory::enumerateFiles(targetSourceDirectory);
    

    std::vector<int4> sourceTetIndices = getSourceTetIndices(tetmeshFilename);

    SimpleMesh* sourceMesh = createMesh(sourceFilename);

    std::vector<SimpleMesh*> targetMeshes;
    for (auto target : targetFiles) {
        targetMeshes.push_back(createMesh(targetSourceDirectory + "/" + target));
    }
    std::cout << "All meshes now in memory" << std::endl;

    CombinedSolverParameters params;
    params.numIter = 15;
    params.nonLinearIter = 10;
    params.linearIter = 250;
    params.useOpt = false;
    params.useOptLM = true;

    CombinedSolver solver(sourceMesh, targetMeshes, sourceTetIndices, params);
    solver.solveAll();
    SimpleMesh* res = solver.result();
    
	if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}
    
    for (SimpleMesh* mesh : targetMeshes) {
        delete mesh;
    }
    delete sourceMesh;

	return 0;
}
