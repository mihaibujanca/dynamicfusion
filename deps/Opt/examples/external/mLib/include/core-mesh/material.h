#ifndef CORE_MESH_MATERIAL_H_
#define CORE_MESH_MATERIAL_H_

namespace ml {

template<class FloatType>
class Material {
public:
	Material() {
		reset();
	}
	Material(Material&& m) {
		swap(*this, m);
	}

	void operator=(Material&& m) {
		swap(*this, m);
	}

	//! adl swap
	friend void swap(Material& a, Material& b) {
		std::swap(a.m_name, b.m_name);
		std::swap(a.m_ambient, b.m_ambient);
		std::swap(a.m_diffuse, b.m_diffuse);
		std::swap(a.m_specular, b.m_specular);
		std::swap(a.m_shiny, b.m_shiny);

		std::swap(a.m_TextureFilename_Ka, b.m_TextureFilename_Ka);
		std::swap(a.m_TextureFilename_Kd, b.m_TextureFilename_Kd);
		std::swap(a.m_TextureFilename_Ks, b.m_TextureFilename_Ks);
		//std::swap(a.m_Texture_Ka, b.m_Texture_Ka);
		//std::swap(a.m_Texture_Kd, b.m_Texture_Kd);
		//std::swap(a.m_Texture_Ks, b.m_Texture_Ks);
	}

	static std::vector<Material> loadFromMTL(const std::string& filename) {
		std::vector<Material> res;
		loadFromMTL(filename, res);
		return res;
	}

	static void loadFromMTL(const std::string& filename, std::vector<Material>& res) {
		res.clear();

		std::ifstream in(filename);
		if (!in.is_open()) throw MLIB_EXCEPTION("could not open file " + filename);
		std::string line;

		Material activeMaterial;	bool found = false;
		while (std::getline(in, line)) {
			std::stringstream ss(line);

			std::string token;
			ss >> token;
			if (token == "newmtl") {
				if (!found) {
					found = true;
				} else {
					res.push_back(activeMaterial);
				}
				activeMaterial.reset();
				ss >> activeMaterial.m_name;
			} else if (token == "Ka") {
				ss >> activeMaterial.m_ambient.x >> activeMaterial.m_ambient.y >> activeMaterial.m_ambient.z;
			} else if (token == "Kd") {
				ss >> activeMaterial.m_diffuse.x >> activeMaterial.m_diffuse.y >> activeMaterial.m_diffuse.z;
			} else if (token == "Ks") {
				ss >> activeMaterial.m_specular.x >> activeMaterial.m_specular.y >> activeMaterial.m_specular.z;
			} else if (token == "Ns") {
				ss >> activeMaterial.m_shiny;
			} else if (token == "map_Ka") {
				ss >> activeMaterial.m_TextureFilename_Ka;
			} else if (token == "map_Kd") {
				ss >> activeMaterial.m_TextureFilename_Kd;
			} else if (token == "map_Ks") {
				ss >> activeMaterial.m_TextureFilename_Ks;
            }
            else if (token == "d") {
                // d token not implemented
				//MLIB_WARNING("d token not implemented");
            }
            else {
				MLIB_WARNING("unknown token: " + line);
			}
		}

		if (found) {
			res.push_back(activeMaterial);
		}
		in.close();

	}

	void reset() {
		m_name = "";
		m_ambient = vec4<FloatType>(0,0,0,0);
		m_diffuse = vec4<FloatType>(0,0,0,0);
		m_specular = vec4<FloatType>(0,0,0,0);
		m_shiny = 0;
		m_TextureFilename_Ka = "";
		m_TextureFilename_Kd = "";
		m_TextureFilename_Ks = "";
		//m_Texture_Ka.free();
		//m_Texture_Kd.free();
		//m_Texture_Ks.free();
	}

	std::string		m_name;
	vec4<FloatType>	m_ambient;
	vec4<FloatType>	m_diffuse;
	vec4<FloatType>	m_specular;
	FloatType		m_shiny;

	std::string			m_TextureFilename_Ka;
	std::string			m_TextureFilename_Kd;
	std::string			m_TextureFilename_Ks;
	//ColorImageR8G8B8A8	m_Texture_Ka;
	//ColorImageR8G8B8A8	m_Texture_Kd;
	//ColorImageR8G8B8A8	m_Texture_Ks;
};

typedef Material<float> Materialf;
typedef Material<double> Materiald;

template<class BinaryDataBuffer, class BinaryDataCompressor, class FloatType>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<< (BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const Material<FloatType> &m) {
    s << m.m_name << m.m_ambient << m.m_diffuse << m.m_specular << m.m_shiny;
    s << m.m_TextureFilename_Ka;
    s << m.m_TextureFilename_Kd;
    s << m.m_TextureFilename_Ks;
    return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class FloatType>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>> (BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, Material<FloatType> &m) {
    s >> m.m_name >> m.m_ambient >> m.m_diffuse >> m.m_specular >> m.m_shiny;
    s >> m.m_TextureFilename_Ka;
    s >> m.m_TextureFilename_Kd;
    s >> m.m_TextureFilename_Ks;
    return s;
}

}

#endif