#include "main.h"
#include "CombinedSolver.h"
#include "../../shared/CombinedSolverParameters.h"
int main(int argc, const char * argv[])
{
    std::string filename = "../data/ye_high2.png";
    if (argc > 1) {
        filename = argv[1];
    }

    ColorImageR8G8B8A8	   image = LodePNG::load(filename);
	ColorImageR32G32B32A32 imageR32(image.getWidth(), image.getHeight());
	for (unsigned int y = 0; y < image.getHeight(); y++) {
		for (unsigned int x = 0; x < image.getWidth(); x++) {
			imageR32(x,y) = image(x,y);
		}
	}
	

    CombinedSolverParameters params;
    params.nonLinearIter = 7;
    params.linearIter = 10;

    CombinedSolver solver(imageR32, params);

	solver.solveAll();

    ColorImageR32G32B32A32* res = solver.getAlbedo();
	ColorImageR8G8B8A8 out(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = math::round(math::clamp(255.0f*(*res)(x, y).x, 0.0f, 255.0f));
			unsigned char g = math::round(math::clamp(255.0f*(*res)(x, y).y, 0.0f, 255.0f));
			unsigned char b = math::round(math::clamp(255.0f*(*res)(x, y).z, 0.0f, 255.0f));
			out(x, y) = vec4uc(r, g, b,255);
		}
	}
	LodePNG::save(out, "outputAlbedo.png");

    res = solver.getShading();
	ColorImageR8G8B8A8 out2(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = math::round(255.0f*math::clamp((*res)(x, y).x, 0.0f, 255.0f));
			unsigned char g = math::round(255.0f*math::clamp((*res)(x, y).y, 0.0f, 255.0f));
			unsigned char b = math::round(255.0f*math::clamp((*res)(x, y).z, 0.0f, 255.0f));
			out2(x, y) = vec4uc(r, g, b, 255);
		}
	}
	LodePNG::save(out2, "outputShading.png");
	return 0;
}
