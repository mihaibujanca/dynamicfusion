
#ifndef CORE_MESH_TRIMESH_INL_H_
#define CORE_MESH_TRIMESH_INL_H_

namespace ml {

	template<class FloatType>
	TriMesh<FloatType>::TriMesh(const MeshData<FloatType>& meshData)
	{
		
		m_vertices.resize(meshData.m_Vertices.size());

		m_bHasNormals = meshData.m_Normals.size() > 0;
		m_bHasColors = meshData.m_Colors.size() > 0;
		m_bHasTexCoords = meshData.m_TextureCoords.size() > 0;

		// Initialize positions (normals, colors and texcoords to be initialized later)
		for (size_t i = 0; i < m_vertices.size(); i++) {
			m_vertices[i].position = meshData.m_Vertices[i];
		}

		for (unsigned int i = 0; i < meshData.m_FaceIndicesVertices.size(); i++) {
			if (meshData.m_FaceIndicesVertices[i].size() == 3) {
				//we need to split vertices if the same vertex has different texcoords and/or normals
				bool bFaceHasNormals = m_bHasNormals && meshData.getFaceIndicesNormals()[i].size() > 0;
				bool bFaceHasTexCoords = m_bHasTexCoords && meshData.getFaceIndicesTexCoords()[i].size() > 0;
				bool bFaceHasColors = m_bHasColors && meshData.getFaceIndicesColors()[i].size() > 0;

				if (bFaceHasNormals || bFaceHasTexCoords || bFaceHasColors) {
					vec3ui coords = vec3ui(0, 0, 0);
					for (unsigned int j = 0; j < 3; j++) {
						Vertex& vert = m_vertices[meshData.getFaceIndicesVertices()[i][j]];
						bool vertexSplit = false;
						if (bFaceHasNormals) { //split if normal is different than the one found before
							const vec3<FloatType>& n = meshData.m_Normals[meshData.getFaceIndicesNormals()[i][j]];
							if (vert.normal != vec3<FloatType>::origin && vert.normal != n) {
								vertexSplit = true;
							}
						}
						if (bFaceHasTexCoords) { //split if texcoord is different than the one found before
							const vec2<FloatType>& t = meshData.m_TextureCoords[meshData.getFaceIndicesTexCoords()[i][j]];
							if (vert.texCoord != vec2<FloatType>::origin && vert.texCoord != t) {
								vertexSplit = true;
							}
						}
						if (bFaceHasColors) { //split if texcoord is different than the one found before
							const vec4<FloatType>& c = meshData.m_Colors[meshData.getFaceIndicesColors()[i][j]];
							if (vert.color != vec4<FloatType>::origin && vert.color != c) {
								vertexSplit = true;
							}
						}

						if (vertexSplit) {
							//MLIB_WARNING("vertex split untested");
							Vertex v = vert;  // copy of vert for splitting out
							if (bFaceHasNormals)	v.normal = meshData.m_Normals[meshData.getFaceIndicesNormals()[i][j]];
							if (bFaceHasTexCoords)	v.texCoord = meshData.m_TextureCoords[meshData.getFaceIndicesTexCoords()[i][j]];
							if (bFaceHasColors)		v.color = meshData.m_Colors[meshData.getFaceIndicesColors()[i][j]];
							m_vertices.push_back(v);
							coords[j] = (unsigned int)m_vertices.size() - 1;
						}
						else {
							if (bFaceHasNormals)	vert.normal = meshData.m_Normals[meshData.getFaceIndicesNormals()[i][j]];
							if (bFaceHasTexCoords)	vert.texCoord = meshData.m_TextureCoords[meshData.getFaceIndicesTexCoords()[i][j]];
							if (bFaceHasColors)		vert.color = meshData.m_Colors[meshData.getFaceIndicesColors()[i][j]];
							coords[j] = meshData.getFaceIndicesVertices()[i][j];
						}
					}
					m_indices.push_back(coords);

					//m_Indices.push_back(vec3ui(meshData.m_FaceIndicesVertices[i][0], meshData.m_FaceIndicesVertices[i][1], meshData.m_FaceIndicesVertices[i][2]));
					//if (hasNormals) {
					//	//we are ignoring the fact that sometimes there might be vertex split required (if a vertex has two different normals)
					//	m_Vertices[m_Indices[i][0]].normal = meshData.m_Normals[meshData.m_FaceIndicesNormals[i][0]];
					//	m_Vertices[m_Indices[i][1]].normal = meshData.m_Normals[meshData.m_FaceIndicesNormals[i][1]];
					//	m_Vertices[m_Indices[i][2]].normal = meshData.m_Normals[meshData.m_FaceIndicesNormals[i][2]];
					//}
					//if (hasTexCoords) {
					//	//we are ignoring the fact that sometimes there might be vertex split required (if a vertex has two different normals)
					//	m_Vertices[m_Indices[i][0]].texCoord = meshData.m_TextureCoords[meshData.m_FaceIndicesTextureCoords[i][0]];
					//	m_Vertices[m_Indices[i][1]].texCoord = meshData.m_TextureCoords[meshData.m_FaceIndicesTextureCoords[i][1]];
					//	m_Vertices[m_Indices[i][2]].texCoord = meshData.m_TextureCoords[meshData.m_FaceIndicesTextureCoords[i][2]];
					//}
				}
				else {
					m_indices.push_back(vec3ui(meshData.m_FaceIndicesVertices[i][0], meshData.m_FaceIndicesVertices[i][1], meshData.m_FaceIndicesVertices[i][2]));
				}
			}
			else {
				MLIB_WARNING("non triangle face found - ignoring it (see meshData for a converter)");
			}
		}
	}


	template<class FloatType>
	void TriMesh<FloatType>::computeNormals() {
		for (size_t i = 0; i < m_vertices.size(); i++) {
			m_vertices[i].normal = vec3<FloatType>::origin;
		}

		for (size_t i = 0; i < m_indices.size(); i++) {
			vec3<FloatType> faceNormal =
				(m_vertices[m_indices[i].y].position - m_vertices[m_indices[i].x].position) ^ (m_vertices[m_indices[i].z].position - m_vertices[m_indices[i].x].position);

			m_vertices[m_indices[i].x].normal += faceNormal;
			m_vertices[m_indices[i].y].normal += faceNormal;
			m_vertices[m_indices[i].z].normal += faceNormal;
		}
		for (size_t i = 0; i < m_vertices.size(); i++) {
			m_vertices[i].normal.normalize();
		}

		m_bHasNormals = true;
	}

	template<class FloatType>
	TriMesh<FloatType> TriMesh<FloatType>::flatLoopSubdivision(unsigned int iterations, float minEdgeLength) const {
		TriMeshf result = *this;
		for (unsigned int i = 0; i < iterations; i++)
			result = result.flatLoopSubdivision(minEdgeLength);
		return result;
	}

	template < class FloatType >
	TriMesh<FloatType> TriMesh<FloatType>::flatten() const {
		TriMesh<FloatType> result;

		size_t index = 0;
		for (const vec3ui& t : m_indices) {
			result.m_vertices.push_back(m_vertices[t[0]]);
			result.m_vertices.push_back(m_vertices[t[1]]);
			result.m_vertices.push_back(m_vertices[t[2]]);

			result.m_indices.push_back(vec3ui(index + 0, index + 1, index + 2));
			index += 3;
		}

		return result;
	}

	template<class FloatType>
	TriMesh<FloatType> TriMesh<FloatType>::flatLoopSubdivision(float minEdgeLength) const
	{
		struct Edge
		{
			Edge(UINT32 _v0, UINT32 _v1)
			{
				v0 = std::min(_v0, _v1);
				v1 = std::max(_v0, _v1);
			}

			union
			{
				struct {
					UINT32 v0, v1;
				};
				UINT64 val;
			};
		};

		struct edgeCompare
		{
			bool operator() (const Edge &a, const Edge &b)
			{
				return a.val < b.val;
			}
		};

		std::map<Edge, UINT, edgeCompare> edgeToNewVertexMap;

		TriMesh<FloatType> result;

		result.m_vertices = m_vertices;
		result.m_indices.reserve(m_indices.size() * 4);

		for (const vec3ui &tri : m_indices)
		{
			/*bool subdivide = true;
			for (UINT eIndex = 0; eIndex < 3; eIndex++)
			{
			const vec3f &v0 = m_Vertices[tri[eIndex]].position;
			const vec3f &v1 = m_Vertices[tri[(eIndex + 1) % 3]].position;
			float edgeLength = vec3f::dist(v0, v1);
			if (edgeLength < minEdgeLength)
			subdivide = false;
			}*/
			bool subdivide = math::triangleArea(m_vertices[tri[0]].position, m_vertices[tri[1]].position, m_vertices[tri[2]].position) >= (minEdgeLength * minEdgeLength);

			if (subdivide)
			{
				UINT edgeMidpoints[3];

				for (UINT eIndex = 0; eIndex < 3; eIndex++)
				{
					const UINT v0 = tri[eIndex];
					const UINT v1 = tri[(eIndex + 1) % 3];
					Edge e = Edge(v0, v1);
					if (edgeToNewVertexMap.count(e) == 0)
					{
						edgeToNewVertexMap[e] = (UINT)result.m_vertices.size();
						result.m_vertices.push_back((m_vertices[v0] + m_vertices[v1]) * (FloatType)0.5);
					}

					edgeMidpoints[eIndex] = edgeToNewVertexMap[e];
				}

				result.m_indices.push_back(vec3ui(tri[0], edgeMidpoints[0], edgeMidpoints[2]));
				result.m_indices.push_back(vec3ui(edgeMidpoints[0], tri[1], edgeMidpoints[1]));
				result.m_indices.push_back(vec3ui(edgeMidpoints[2], edgeMidpoints[1], tri[2]));
				result.m_indices.push_back(vec3ui(edgeMidpoints[2], edgeMidpoints[0], edgeMidpoints[1]));
			}
			else
			{
				result.m_indices.push_back(tri);
			}
		}

		return result;
	}


}

#endif // CORE_MESH_TRIMESH_INL_H_