#ifndef CORE_MESH_POINTCLOUD_H_
#define CORE_MESH_POINTCLOUD_H_

namespace ml {

template <class FloatType>
class PointCloud {
public:
	PointCloud() {}

	PointCloud(const std::vector < vec3<FloatType>>& points) {
		m_points = points;
	}

	//!conversion from a binary voxel grid
	PointCloud(const BinaryGrid3& grid, FloatType voxelSize = (FloatType)1) {
		for (unsigned int z = 0; z < grid.getDimZ(); z++) {
			for (unsigned int y = 0; y < grid.getDimY(); y++) {
				for (unsigned int x = 0; x < grid.getDimX(); x++) {
					if (grid.isVoxelSet(x,y,z)) {
						vec3<FloatType> p((FloatType)x,(FloatType)y,(FloatType)z);
						m_points.push_back(p * voxelSize);
					}
				}
			}
		}
	}

	PointCloud(PointCloud&& pc) {
		m_points = std::move(pc.m_points);
		m_normals = std::move(pc.m_normals);
		m_colors = std::move(pc.m_colors);
		m_texCoords = std::move(pc.m_texCoords);
	}
	void operator=(PointCloud&& pc) {
		m_points = std::move(pc.m_points);
		m_normals = std::move(pc.m_normals);
		m_colors = std::move(pc.m_colors);
		m_texCoords = std::move(pc.m_texCoords);
	}

	bool hasNormals() const { return m_normals.size() > 0; }
	bool hasColors() const { return m_colors.size() > 0; }
	bool hasTexCoords() const { return m_texCoords.size() > 0; }

	void clear() {
		m_points.clear();
		m_normals.clear();
		m_colors.clear();
		m_texCoords.clear();
	}

	bool isConsistent() const {
		bool is = true;
		if (m_normals.size() > 0 && m_normals.size() != m_points.size())		is = false;
		if (m_colors.size() > 0 && m_colors.size() != m_points.size())			is = false;
		if (m_texCoords.size() > 0 && m_texCoords.size() != m_points.size())	is = false;
		return is;
	}

	bool isEmpty() const {
		return m_points.size() == 0;
	}
	
	
	//! merges two point clouds (assumes the same memory layout/type)
	void merge(const PointCloud<FloatType>& other) {

		if (other.isEmpty()) {
			return;
		}
		if (isEmpty()) {
			*this = other;
			return;
		}

		
		if (hasNormals() != other.hasNormals()) {
			MLIB_WARNING("normals deleted");
			m_normals.clear();
		}
		if (hasColors() != other.hasColors()) {
			MLIB_WARNING("colors deleted");
			m_colors.clear();
		}
		if (hasTexCoords() != other.hasTexCoords()) {
			MLIB_WARNING("texcoords deleted");
			m_texCoords.clear();
		}

		
		size_t vertsBefore = m_points.size();
		size_t normsBefore = m_normals.size();
		size_t colorBefore = m_colors.size();
		size_t texCoordsBefore = m_texCoords.size();
		m_points.insert(m_points.end(), other.m_points.begin(), other.m_points.end());

		if (hasColors() || isEmpty())		m_colors.insert(m_colors.end(), other.m_colors.begin(), other.m_colors.end());
		if (hasNormals() || isEmpty())		m_normals.insert(m_normals.end(), other.m_normals.begin(), other.m_normals.end());
		if (hasTexCoords() || isEmpty())	m_texCoords.insert(m_texCoords.end(), other.m_texCoords.begin(), other.m_texCoords.end());
		
	}
	

	void applyTransform(const Matrix4x4<FloatType>& t) {
		for (size_t i = 0; i < m_points.size(); i++) {
			m_points[i] = t*m_points[i];
		}
		Matrix4x4<FloatType> invTrans = t.getInverse().getTranspose();
		for (size_t i = 0; i < m_normals.size(); i++) {
			m_normals[i] = invTrans.transformNormalAffine(m_normals[i]);
			m_normals[i].normalizeIfNonzero();
		}
	}

    //! Computes the bounding box of the mesh (not cached!)
    BoundingBox3<FloatType> computeBoundingBox() const {
        BoundingBox3<FloatType> bb;
        for (size_t i = 0; i < m_points.size(); i++) {
            bb.include(m_points[i]);
        }
        return bb;
    }


	void sparsify(size_t newMaxCount) {
		std::default_random_engine generator;

		while (m_points.size() > newMaxCount) {
			std::uniform_int_distribution<size_t> distribution(0, m_points.size() - 1);
			size_t n = distribution(generator);

			std::swap(m_points[n], m_points.back());
			m_points.pop_back();

			if (hasColors()) {
				std::swap(m_colors[n], m_colors.back());
				m_colors.pop_back();
			}
			if (hasNormals()) {
				std::swap(m_normals[n], m_normals.back());
				m_normals.pop_back();
			}
			if (hasTexCoords()) {
				std::swap(m_texCoords[n], m_texCoords.back());
				m_texCoords.pop_back();
			}
		}
	}

	size_t sparsifyUniform(FloatType thresh, bool approx);
	size_t sparsifyUniform(unsigned int targetNumVerts, bool approx = true, FloatType startThresh = 0.005f /*5mm*/) {
		FloatType thresh = startThresh;
		while (m_points.size() > targetNumVerts) {
			sparsifyUniform(thresh, approx);
			thresh *= 2.0f;
		}
		return m_points.size();
	}


	std::vector<vec3<FloatType>> m_points;
	std::vector<vec3<FloatType>> m_normals;
	std::vector<vec4<FloatType>> m_colors;
	std::vector<vec2<FloatType>> m_texCoords;
private:

	inline vec3i toVirtualVoxelPos(const vec3<FloatType>& v, FloatType voxelSize) {
		return vec3i(v / voxelSize + (FloatType)0.5*vec3<FloatType>(math::sign(v)));
	}
	//! returns -1 if there is no vertex closer to 'v' than thresh; otherwise the vertex id of the closer vertex is returned
	size_t hasNearestNeighbor(const vec3i& coord, SparseGrid3<std::list<std::pair<vec3<FloatType>, size_t> > > &neighborQuery, const vec3<FloatType>& v, FloatType thresh);

	//! returns -1 if there is no vertex closer to 'v' than thresh; otherwise the vertex id of the closer vertex is returned (manhattan distance)
	size_t hasNearestNeighborApprox(const vec3i& coord, SparseGrid3<size_t> &neighborQuery, FloatType thresh);
};

typedef PointCloud<float>	PointCloudf;
typedef PointCloud<double>	PointCloudd;

} // namespace ml

#include "pointCloud.cpp"

#endif