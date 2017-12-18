
#ifndef CORE_MESH_POINTIO_INL_H_
#define CORE_MESH_POINTIO_INL_H_

namespace ml {



	template <class FloatType>
	void PointCloudIO<FloatType>::loadFromPLY( const std::string& filename, PointCloud<FloatType>& pc )
	{
		std::ifstream file(filename, std::ios::binary);
		if (!file.is_open())	throw MLIB_EXCEPTION("Could not open file " + filename);			

		PlyHeader header(file);

		if (header.m_numVertices == (unsigned int)-1) throw MLIB_EXCEPTION("no vertices found");

		pc.m_points.resize(header.m_numVertices);
		if (header.m_bHasNormals)	pc.m_normals.resize(header.m_numVertices);
		if (header.m_bHasColors)	pc.m_colors.resize(header.m_numVertices);

		if (header.m_bBinary) {
			unsigned int size = 0;
			for (size_t i = 0; i < header.m_properties["vertex"].size(); i++) {
				size += header.m_properties["vertex"][i].byteSize;
			}
			char* data = new char[size*header.m_numVertices];
			file.read(data, size*header.m_numVertices);

			for (unsigned int i = 0; i < header.m_numVertices; i++) {
				unsigned int byteOffset = 0;
				const std::vector<PlyHeader::PlyProperty>& vertexProperties = header.m_properties["vertex"];

				for (unsigned int j = 0; j < vertexProperties.size(); j++) {

					auto readNextDecimalValue = [&]()
					{
						int byteSize = vertexProperties[j].byteSize;
						FloatType f;
						if (byteSize == 4)
						{
							f = (FloatType)((float*)&data[i*size + byteOffset])[0];
						}
						else if (byteSize == 8)
						{
							f = (FloatType)((double*)&data[i*size + byteOffset])[0];
						}
						byteOffset += byteSize;
						return f;
					};

					if (vertexProperties[j].name == "x") {
						pc.m_points[i].x = readNextDecimalValue();
						//pc.m_points[i].x = ((float*)&data[i*size + byteOffset])[0];
						//byteOffset += vertexProperties[j].byteSize;
					}
					else if (vertexProperties[j].name == "y") {
						pc.m_points[i].y = readNextDecimalValue();
						//pc.m_points[i].y = ((float*)&data[i*size + byteOffset])[0];
						//byteOffset += vertexProperties[j].byteSize;
					}
					else if (vertexProperties[j].name == "z") {
						pc.m_points[i].z = readNextDecimalValue();
						//pc.m_points[i].z = ((float*)&data[i*size + byteOffset])[0];
						//byteOffset += vertexProperties[j].byteSize;
					}
					else if (vertexProperties[j].name == "nx") {
						pc.m_normals[i].x = ((float*)&data[i*size + byteOffset])[0];
						byteOffset += vertexProperties[j].byteSize;
					}
					else if (vertexProperties[j].name == "ny") {
						pc.m_normals[i].y = ((float*)&data[i*size + byteOffset])[0];
						byteOffset += vertexProperties[j].byteSize;
					}
					else if (vertexProperties[j].name == "nz") {
						pc.m_normals[i].z = ((float*)&data[i*size + byteOffset])[0];
						byteOffset += vertexProperties[j].byteSize;
					}
					else if (vertexProperties[j].name == "red") {
						pc.m_colors[i].x = ((unsigned char*)&data[i*size + byteOffset])[0];	pc.m_colors[i].x/=255.0f;
						byteOffset += vertexProperties[j].byteSize;
					}
					else if (vertexProperties[j].name == "green") {
						pc.m_colors[i].y = ((unsigned char*)&data[i*size + byteOffset])[0];	pc.m_colors[i].y/=255.0f;
						byteOffset += vertexProperties[j].byteSize;
					}
					else if (vertexProperties[j].name == "blue") {
						pc.m_colors[i].z = ((unsigned char*)&data[i*size + byteOffset])[0];	pc.m_colors[i].z/=255.0f;
						byteOffset += vertexProperties[j].byteSize;
					}
					else if (vertexProperties[j].name == "alpha") {
						pc.m_colors[i].w = ((unsigned char*)&data[i*size + byteOffset])[0];	pc.m_colors[i].w/=255.0f;
						byteOffset += vertexProperties[j].byteSize;
					} else {
						//unknown (ignore)
						byteOffset += vertexProperties[j].byteSize;
					}

				}
				assert(byteOffset == size);

			}	

			delete [] data;
		} else {
			MLIB_WARNING("untested");
			for (size_t i = 0; i < header.m_numVertices; i++) {
				std::string line;
				std::getline(file, line);
				std::stringstream ss(line); // TODO replace with sscanf which should be faster
				ss >> pc.m_points[i].x >> pc.m_points[i].y >> pc.m_points[i].z;
				if (header.m_bHasColors) {
					ss >> pc.m_colors[i].x >> pc.m_colors[i].y >> pc.m_colors[i].z;
					pc.m_colors[i] /= (FloatType)255.0;
				}
			}
		}

		file.close();
	}

} // namespace ml

#endif