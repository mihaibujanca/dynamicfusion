#ifndef __LANDMARKSET__H__
#define __LANDMARKSET__H__

#include "LandMark.h"

#include <vector>
#include <iostream>
#include <fstream>

class LandMarkSet : public std::vector<LandMark>
{
	public:
		
		void loadFromFile(const char* filename)
		{
			this->clear();

			std::ifstream in(filename, std::fstream::in);

			if(!in.good())
			{
				std::cout << "Could not open marker file " << filename << std::endl;
				assert(false);
			}

			unsigned int nMarkers;
			in >> nMarkers;

			for(unsigned int m = 0; m<nMarkers; m++)
			{
				std::vector<float> position; position.resize(3);
				in >> position[0];
				in >> position[1];
				in >> position[2];

				float radius;
				in >> radius;

				unsigned int vertexIndex;
				in >> vertexIndex;

				push_back(LandMark(position, vertexIndex, radius));
			}

			in.close();
		}
};

#endif //__LANDMARKSET__H__
