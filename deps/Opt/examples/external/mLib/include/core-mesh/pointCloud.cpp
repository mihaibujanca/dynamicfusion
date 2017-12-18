
#ifndef CORE_MESH_POINTCLOUD_INL_H_
#define CORE_MESH_POINTCLOUD_INL_H_

namespace ml {
template<class FloatType>
std::ostream& operator<<(std::ostream& os, const PointCloud<FloatType>& pointCloud) {
	os << "MeshData:\n"
		<< "\tVertices:  " << pointCloud.m_points.size() << "\n"
		<< "\tColors:    " << pointCloud.m_colors.size() << "\n"
		<< "\tNormals:   " << pointCloud.m_normals.size() << "\n"
		<< std::endl;

	return os;
}





template <class FloatType>
size_t PointCloud<FloatType>::hasNearestNeighbor(const vec3i& coord, SparseGrid3<std::list<std::pair<vec3<FloatType>, size_t> > > &neighborQuery, const vec3<FloatType>& v, FloatType thresh)
{
	FloatType threshSq = thresh*thresh;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				vec3i c = coord + vec3i(i, j, k);
				if (neighborQuery.exists(c)) {
					for (const std::pair<vec3<FloatType>, size_t>& n : neighborQuery[c]) {
						if (vec3<FloatType>::distSq(v, n.first) < threshSq) {
							return n.second;
						}
					}
				}
			}
		}
	}
	return (size_t)-1;
}

template <class FloatType>
size_t PointCloud<FloatType>::hasNearestNeighborApprox(const vec3i& coord, SparseGrid3<size_t> &neighborQuery, FloatType thresh) {
	FloatType threshSq = thresh*thresh;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				vec3i c = coord + vec3i(i, j, k);
				if (neighborQuery.exists(c)) {
					return neighborQuery[c];
				}
			}
		}
	}
	return (size_t)-1;
}




template <class FloatType>
size_t PointCloud<FloatType>::sparsifyUniform(FloatType thresh, bool approx)
{
	if (thresh <= (FloatType)0)	throw MLIB_EXCEPTION("invalid thresh " + std::to_string(thresh));
	size_t numV = m_points.size();

	std::vector<size_t> vertexLookUp;	vertexLookUp.resize(numV);
	std::vector<vec3<FloatType>> new_verts; new_verts.reserve(numV);
	std::vector<vec4<FloatType>> new_color;		if (hasColors())		new_color.reserve(m_colors.size());
	std::vector<vec3<FloatType>> new_normals;	if (hasNormals())		new_normals.reserve(m_normals.size());
	std::vector<vec2<FloatType>> new_tex;		if (hasTexCoords())		new_tex.reserve(m_texCoords.size());

	size_t cnt = 0;
	if (approx) {
		SparseGrid3<size_t> neighborQuery(0.6f, numV * 2);
		for (size_t v = 0; v < numV; v++) {

			const vec3<FloatType>& vert = m_points[v];
			vec3i coord = toVirtualVoxelPos(vert, thresh);
			size_t nn = hasNearestNeighborApprox(coord, neighborQuery, thresh);

			if (nn == (size_t)-1) {
				neighborQuery[coord] = cnt;
				new_verts.push_back(vert);
				vertexLookUp[v] = cnt;
				cnt++;
				if (hasColors())	new_color.push_back(m_colors[v]);
				if (hasNormals())	new_normals.push_back(m_normals[v]);
				if (hasTexCoords())	new_tex.push_back(m_texCoords[v]);
			}
			else {
				vertexLookUp[v] = nn;
			}
		}
	}
	else {
		SparseGrid3<std::list<std::pair<vec3<FloatType>, size_t> > > neighborQuery(0.6f, numV * 2);
		for (size_t v = 0; v < numV; v++) {

			const vec3<FloatType>& vert = m_points[v];
			vec3i coord = toVirtualVoxelPos(vert, thresh);
			size_t nn = hasNearestNeighbor(coord, neighborQuery, vert, thresh);

			if (nn == (size_t)-1) {
				neighborQuery[coord].push_back(std::make_pair(vert, cnt));
				new_verts.push_back(vert);
				vertexLookUp[v] = cnt;
				cnt++;
				if (hasColors())	new_color.push_back(m_colors[v]);
				if (hasNormals())	new_normals.push_back(m_normals[v]);
				if (hasTexCoords())	new_tex.push_back(m_texCoords[v]);
			}
			else {
				vertexLookUp[v] = nn;
			}
		}
	}

	if (m_points.size() != new_verts.size()) {
		m_points = std::vector<vec3<FloatType>>(new_verts.begin(), new_verts.end());

		if (hasColors())	m_colors = std::vector<vec4<FloatType>>(new_color.begin(), new_color.end());
		if (hasNormals())	m_normals = std::vector<vec3<FloatType>>(new_normals.begin(), new_normals.end());
		if (hasTexCoords())	m_texCoords = std::vector<vec2<FloatType>>(new_tex.begin(), new_tex.end());
	}

	return cnt;
}





}

#endif