
#ifndef CORE_MESH_PLYHEADER_H_
#define CORE_MESH_PLYHEADER_H_

namespace ml {

	struct PlyHeader {
		struct PlyProperty {
			PlyProperty() {
				byteSize = 0;
			}
			std::string name;
			unsigned int byteSize;
		};
		PlyHeader(std::ifstream& file) {
			m_numVertices = (unsigned int)-1;
			m_numFaces = (unsigned int)-1;
			m_bHasNormals = false;
			m_bHasColors = false;

			read(file);
		}
		PlyHeader() {
			m_numVertices = (unsigned int)-1;
			m_numFaces = (unsigned int)-1;
			m_bHasNormals = false;
			m_bHasColors = false;
		}
		unsigned int m_numVertices;
		unsigned int m_numFaces;
		std::map<std::string, std::vector<PlyProperty>> m_properties;
		bool m_bBinary;
		bool m_bHasNormals;
		bool m_bHasColors;

		void read(std::ifstream& file) {
			std::string activeElement = "";
			std::string line;
			util::safeGetline(file, line);
			while (line.find("end_header") == std::string::npos) {
				PlyHeaderLine(line, *this, activeElement);
				util::safeGetline(file, line);
			}
		}

		static void PlyHeaderLine(const std::string& line, PlyHeader& header, std::string& activeElement) {

			std::stringstream ss(line);
			std::string currWord;
			ss >> currWord;

	
			if (currWord == "element") {
				ss >> currWord;
				activeElement = currWord;
				if (currWord == "vertex") {
					ss >> header.m_numVertices;
				} else if (currWord == "face") {
					ss >> header.m_numFaces;
				}
			} 
			else if (currWord == "format") {
				ss >> currWord;
				if (currWord == "binary_little_endian")	{
					header.m_bBinary = true;
				} else {
					header.m_bBinary = false;
				}
			}
			else if (currWord == "property") {
				if (!util::endsWith(line, "vertex_indices") && !util::endsWith(line, "vertex_index")) {
					PlyHeader::PlyProperty p;
					std::string which;
					ss >> which;
					ss >> p.name;
					if (activeElement == "vertex") {
						if (p.name == "nx")	header.m_bHasNormals = true;
						if (p.name == "red") header.m_bHasColors = true;
					}

					if (which == "double") p.byteSize = 8;
					else if (which == "float" || which == "int" || which == "uint") p.byteSize = 4;
					else if (which == "uchar" || which == "char") p.byteSize = 1;
					else {
						throw MLIB_EXCEPTION("unkown data type");
					}
					header.m_properties[activeElement].push_back(p);
				}
				else {
					//property belonging to unknown element
				}
			}
		}
	};


} // namespace ml

#endif