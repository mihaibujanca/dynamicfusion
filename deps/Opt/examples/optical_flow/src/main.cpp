#include "mLibInclude.h"
#include "CombinedSolver.h"
#include "ImageHelper.h"


void renderFlowVecotors(ColorImageR8G8B8A8& image, const BaseImage<float2>& flowVectors) {
	const unsigned int skip = 5;	//only every n-th pixel
	const float lengthRed = 5.0f;
	
	for (unsigned int j = 1; j < image.getHeight() - 1; j += skip) {
		for (unsigned int i = 1; i < image.getWidth() - 1; i += skip) {
			
			const float2& flowVector = flowVectors(i, j);
			vec2i start = vec2i(i, j);
			vec2i end = start + vec2i(math::round(flowVector.x), math::round(flowVector.y));
			float len = vec2f(flowVector.x, flowVector.y).length();
			vec4uc color = math::round(255.0f*BaseImageHelper::convertDepthToRGBA(len, 0.0f, 5.0f)*2.0f);	color.w = 255;
			//vec4uc color = math::round(255.0f*vec4f(0.1f, 0.8f, 0.1f, 1.0f));	//TODO color-code length

			ImageHelper::drawLine(image, start, end, color);
		}
	}
}

int main(int argc, const char * argv[]) {

    std::string srcFile = "../data/dogdance0.png";
    std::string tarFile = "../data/dogdance1.png";
    if (argc > 1) {
        assert(argc > 2);
        srcFile = argv[1];
        tarFile = argv[2];
    }

	ColorImageR8G8B8A8 imageSrc = LodePNG::load(srcFile);
	ColorImageR8G8B8A8 imageTar = LodePNG::load(tarFile);

	ColorImageR32 imageSrcGray = imageSrc.convertToGrayscale();
	ColorImageR32 imageTarGray = imageTar.convertToGrayscale();

    CombinedSolverParameters params;
    params.numIter = 3;
    params.nonLinearIter = 1;
    params.linearIter = 50;

    CombinedSolver solver(imageSrcGray, imageTarGray, params);
    solver.solveAll();
    BaseImage<float2> flowVectors = solver.result();

	const std::string outFile = "out.png";
	ColorImageR8G8B8A8 out = imageSrc;
	renderFlowVecotors(out, flowVectors);
	LodePNG::save(out, outFile);

	return 0;
}
