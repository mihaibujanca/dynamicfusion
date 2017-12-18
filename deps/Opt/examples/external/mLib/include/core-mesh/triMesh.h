#ifndef CORE_MESH_TRIMESH_H_
#define CORE_MESH_TRIMESH_H_

namespace ml {

	template<class FloatType>
	class TriMesh
	{
	public:

		// ********************************
		// Vertex class of the Tri Mesh
		// ********************************
		class Vertex {
		public:
			Vertex() : position(vec3<FloatType>::origin), normal(vec3<FloatType>::origin), color(vec4<FloatType>::origin), texCoord(vec2<FloatType>::origin) { }
			Vertex(const vec3<FloatType>& _position) : position(_position) { }
			Vertex(const vec3<FloatType>& _p, const vec3<FloatType>& _n) : position(_p), normal(_n) { }
			Vertex(const vec3<FloatType>& _p, const vec3<FloatType>& _n, const vec4<FloatType>& _c, const vec2<FloatType>& _t) : position(_p), normal(_n), color(_c), texCoord(_t) { }

			Vertex operator*(FloatType t) const {
				return Vertex(position*t, normal*t, color*t, texCoord*t);
			}
			Vertex operator/(FloatType t) const {
				return Vertex(position/t, normal/t, color/t, texCoord/t);
			}
			Vertex operator+(const Vertex& other) const {
				return Vertex(position+other.position, normal+other.normal, color+other.color, texCoord+other.texCoord);
			}
			Vertex operator-(const Vertex& other) const {
				return Vertex(position-other.position, normal-other.normal, color-other.color, texCoord-other.texCoord);
			}

			void operator*=(FloatType t) {
				*this = *this * t;
			}
			void operator/=(FloatType t) const {
				*this = *this / t;
			}
			void operator+=(const Vertex& other) const {
				*this = *this + other;
			}
			void operator-=(const Vertex& other) const {
				*this = *this - other;
			}

			//
			// If you change the order of these fields, you must update the layout fields in D3D11TriMesh::layout 
			//
			vec3<FloatType> position;
			vec3<FloatType> normal;
			vec4<FloatType> color;
			vec2<FloatType> texCoord;
		private:
		};


		// ********************************
		// Triangle class of the Tri Mesh
		// ********************************
		class Triangle {
		public:

			Triangle(const Vertex *v0, const Vertex *v1, const Vertex *v2, unsigned int triIdx = 0, unsigned int meshIdx = 0) {
				assert (v0 && v1 && v2);
				this->v0 = v0;
				this->v1 = v1;
				this->v2 = v2;
				m_Center = (v0->position + v1->position + v2->position)/(FloatType)3.0;
				m_TriangleIndex = triIdx;
				m_MeshIndex = meshIdx;
			}


			Vertex getSurfaceVertex(FloatType u, FloatType v) const {
				return *v0 *((FloatType)1.0 - u - v) + *v1 *u + *v2 *v;
			}
			vec3<FloatType> getSurfacePosition(FloatType u, FloatType v) const 	{
				return v0->position*((FloatType)1.0 - u - v) + v1->position*u + v2->position*v;
			}
			vec4<FloatType> getSurfaceColor(FloatType u, FloatType v) const {
				return v0->color*((FloatType)1.0 - u - v) + v1->color*u + v2->color*v;
			}
			vec3<FloatType> getSurfaceNormal(FloatType u, FloatType v) const {
				return v0->normal*((FloatType)1.0 - u - v) + v1->normal*u + v2->normal*v;
			}
			vec2<FloatType> getSurfaceTexCoord(FloatType u, FloatType v) const {
				return v0->texCoord*((FloatType)1.0 - u - v) + v1->texCoord*u + v2->texCoord*v;
			}

			bool intersect(const Ray<FloatType> &r, FloatType& _t, FloatType& _u, FloatType& _v, FloatType tmin = (FloatType)0, FloatType tmax = std::numeric_limits<FloatType>::max(), bool intersectOnlyFrontFaces = false) const {
				return intersection::intersectRayTriangle(v0->position, v1->position, v2->position, r, _t, _u, _v, tmin, tmax, intersectOnlyFrontFaces);
			}

			bool intersects(const Triangle& other) const {
				return intersection::intersectTriangleTriangle(v0->position,v1->position,v2->position, other.v0->position,other.v1->position,other.v2->position);
			}

			void includeInBoundingBox(BoundingBox3<FloatType> &bb) const {
				bb.include(v0->position);
				bb.include(v1->position);
				bb.include(v2->position);
			}

            BoundingBox3<FloatType> computeBoundingBox() const {
				BoundingBox3<FloatType> bb;
				includeInBoundingBox(bb);
				return bb;
			}

			const vec3<FloatType>& getCenter() const {
				return m_Center;
			}

			const Vertex& getV0() const {
				return *v0;
			}
			const Vertex& getV1() const {
				return *v1;
			}
			const Vertex& getV2() const {
				return *v2;
			}

			unsigned int getIndex() const {
				return m_TriangleIndex;
			}
			unsigned int getMeshIndex() const {
				return m_MeshIndex;
			}

		private:
			const Vertex *v0, *v1, *v2;			
			vec3<FloatType> m_Center;	//TODO check if we want to store the center
			unsigned int m_TriangleIndex;	//! 0-based triangle index within it's mesh 
			unsigned int m_MeshIndex;		//! 0-based mesh index; used for accelerators that take an std::vector of triMeshes 
		};


		// ********************************
		// TriMesh itself
		// ********************************
		TriMesh() : m_vertices(), m_indices() {
			m_bHasNormals = false;
			m_bHasTexCoords = false;
			m_bHasColors = false;
		}
		TriMesh(const MeshData<FloatType>& meshData);

		TriMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices, const bool recomputeNormals = false,
			const bool hasNormals = false, const bool hasTexCoords = false, const bool hasColors = false) {
				if (indices.size()%3 != 0)	throw MLIB_EXCEPTION("not a tri mesh");
				m_vertices = vertices;
				m_indices.resize(indices.size()/3);
				memcpy(&m_indices[0], &indices[0], indices.size()*sizeof(unsigned int));
				m_bHasNormals = hasNormals;
				m_bHasTexCoords = hasTexCoords;
				m_bHasColors = hasColors;
				if (recomputeNormals) {
					computeNormals();
				}
		}

		TriMesh(const std::vector<Vertex>& vertices, const std::vector<vec3ui>& indices, bool recomputeNormals = false,
			const bool hasNormals = false, const bool hasTexCoords = false, const bool hasColors = false) {
				m_vertices = vertices;
				m_indices = indices;
				m_bHasNormals = hasNormals;
				m_bHasTexCoords = hasTexCoords;
				m_bHasColors = hasColors;
				if (recomputeNormals) {
					computeNormals();
				}
		}

		TriMesh(
			const std::vector<vec3<FloatType>>& vertices, 
			const std::vector<unsigned int>& indices, 
			const std::vector<vec4<FloatType>>& colors,
			const std::vector<vec3<FloatType>>& normals,
			const std::vector<vec2<FloatType>>& texCoords) :
		TriMesh(vertices.size(), indices.size(), 
			&vertices[0], &indices[0],
			colors.size() > 0 ? &colors[0] : nullptr,
			normals.size() > 0 ? &normals[0] : nullptr,
			texCoords.size() > 0 ? &texCoords[0] : nullptr) 
		{
		}

		TriMesh(const std::vector<vec3<FloatType> >& vertices, const std::vector<unsigned int>& indices) : TriMesh(vertices.size(), indices.size(), &vertices[0], &indices[0]) {}

		TriMesh(
			size_t numVertices, size_t numIndices,
			const vec3<FloatType>* vertices, 
			const unsigned int* indices, 
			const vec4<FloatType>* colors = nullptr, 
			const vec3<FloatType>* normals = nullptr, 
			const vec2<FloatType>* texCoords = nullptr) 
		{
			if (numIndices % 3 != 0) throw MLIB_EXCEPTION("not a tri mesh");
			m_bHasColors = colors != nullptr;
			m_bHasNormals = normals != nullptr;
			m_bHasTexCoords = texCoords != nullptr;
			m_vertices.resize(numVertices);
			for (size_t i = 0; i < numVertices; i++) {
				m_vertices[i].position = vertices[i];
				if (colors) m_vertices[i].color = colors[i];
				if (normals) m_vertices[i].normal = normals[i];
				if (texCoords) m_vertices[i].texCoord = texCoords[i];
			}
			m_indices.resize(numIndices/3);
			for (size_t i = 0; i < numIndices/3; i++) {
				m_indices[i] = vec3ui(indices[3*i+0],indices[3*i+1],indices[3*i+2]);
			}
		}

		TriMesh(const BoundingBox3<FloatType>& bbox, const vec4<FloatType>& color = vec4<FloatType>(1.0,1.0,1.0,1.0)) {
			std::vector<vec3<FloatType>> vertices;
			std::vector<vec3ui> indices;
			std::vector<vec3<FloatType>> normals;
			bbox.makeTriMesh(vertices, indices, normals);

			m_vertices.resize(vertices.size());
			for (size_t i = 0; i < vertices.size(); i++) {
				m_vertices[i].color = color;
				m_vertices[i].position = vertices[i];
				m_vertices[i].normal = normals[i];
			}
			m_indices = indices;
			m_bHasColors = m_bHasNormals = m_bHasTexCoords = true;
		}

		TriMesh(const TriMesh& other) {
			m_vertices = other.m_vertices;
			m_indices = other.m_indices;
			m_bHasNormals = other.m_bHasNormals;
			m_bHasTexCoords = other.m_bHasTexCoords;
			m_bHasColors = other.m_bHasColors;
		}

		TriMesh(TriMesh&& t) {
			swap(*this, t);
		}


		TriMesh(const BinaryGrid3& grid, Matrix4x4<FloatType> voxelToWorld = Matrix4x4<FloatType>::identity(), bool withNormals = false, const vec4<FloatType>& color = vec4<FloatType>(0.5,0.5,0.5,0.5)) {
			// Pre-allocate space
			size_t nVoxels = grid.getNumOccupiedEntries();
			size_t nVertices = (withNormals)? nVoxels*24 : nVoxels*8;
			size_t nIndices = nVoxels*12;
			m_vertices.reserve(nVertices);
			m_indices.reserve(nIndices);
			// Temporaries
			vec3<FloatType> verts[24];
			vec3ui indices[12];
			vec3<FloatType> normals[24];
			for (size_t z = 0; z < grid.getDimZ(); z++) {
				for (size_t y = 0; y < grid.getDimY(); y++) {
					for (size_t x = 0; x < grid.getDimX(); x++) {
						if (grid.isVoxelSet(x,y,z)) {
							vec3<FloatType> p((FloatType)x,(FloatType)y,(FloatType)z);
							vec3<FloatType> pMin = p - (FloatType)0.5;
							vec3<FloatType> pMax = p + (FloatType)0.5;
							p = voxelToWorld * p;

							BoundingBox3<FloatType> bb;
							bb.include(voxelToWorld * pMin);
							bb.include(voxelToWorld * pMax);

							if (withNormals) {
								bb.makeTriMesh(verts,indices,normals);

								unsigned int vertIdxBase = static_cast<unsigned int>(m_vertices.size());
								for (size_t i = 0; i < 24; i++) {
									m_vertices.emplace_back(verts[i], normals[i]);
								}
								for (size_t i = 0; i < 12; i++) {
									indices[i] += vertIdxBase;
									m_indices.push_back(indices[i]);
								}
							} else {
								bb.makeTriMesh(verts, indices);

								unsigned int vertIdxBase = static_cast<unsigned int>(m_vertices.size());
								for (size_t i = 0; i < 8; i++) {
									m_vertices.emplace_back(verts[i]);
								}
								for (size_t i = 0; i < 12; i++) {
									indices[i] += vertIdxBase;
									m_indices.push_back(indices[i]);
								}
							}
						}
					}
				}
			}
			for (size_t i = 0; i < m_vertices.size(); i++) {
				m_vertices[i].color = color;
			}
			m_bHasNormals = withNormals;
			m_bHasTexCoords = false;
			m_bHasColors = true;
		}

		~TriMesh() {
		}

		//! assignment operator
		void operator=(const TriMesh& other) {
			m_vertices = other.m_vertices;
			m_indices = other.m_indices;
			m_bHasNormals = other.m_bHasNormals;
			m_bHasTexCoords = other.m_bHasTexCoords;
			m_bHasColors = other.m_bHasColors;
		}

		//! move operator
		void operator=(TriMesh&& t) {
			std::swap(*this, t);
		}

		//! adl swap
		friend void swap(TriMesh& a, TriMesh& b) {
			std::swap(a.m_vertices, b.m_vertices);
			std::swap(a.m_indices, b.m_indices);
			std::swap(a.m_bHasNormals, b.m_bHasNormals);
			std::swap(a.m_bHasTexCoords, b.m_bHasTexCoords);
			std::swap(a.m_bHasColors, b.m_bHasColors);
		}

		void clear() {
			m_vertices.clear();
			m_indices.clear();
			m_bHasNormals = false;
			m_bHasTexCoords = false;
			m_bHasColors = false;
		}
		bool empty() const {
			return m_vertices.empty();
		}

		void transform(const Matrix4x4<FloatType>& m) {
			Matrix4x4<FloatType> invTrans = m.getInverse().getTranspose();
			for (Vertex& v : m_vertices) {
				v.position = m * v.position;
				v.normal = invTrans.transformNormalAffine(v.normal);
				v.normal.normalizeIfNonzero();
			}
		}

		void scale(FloatType s) { scale(vec3<FloatType>(s, s, s)); }

		void scale(const vec3<FloatType>& v) {
			for (Vertex& mv : m_vertices) for (UINT i = 0; i < 3; i++) { mv.position[i] *= v[i]; }
		}

		//! overwrites/sets the mesh color
		void setColor(const vec4<FloatType>& c) {
			for (auto& v : m_vertices) {
				v.color = c;
			}
		}

		//! Computes the bounding box of the mesh (not cached!)
		BoundingBox3<FloatType> computeBoundingBox() const {
			BoundingBox3<FloatType> bb;
			for (size_t i = 0; i < m_vertices.size(); i++) {
				bb.include(m_vertices[i].position);
			}
			return bb;
		}

		//! Computes the vertex normals of the mesh
		void computeNormals();

        //! Creates a flat Loop-subdivision of the mesh
        TriMesh<FloatType> flatLoopSubdivision(float minEdgeLength) const;
        TriMesh<FloatType> flatLoopSubdivision(UINT iterations, float minEdgeLength) const;

        TriMesh<FloatType> flatten() const;

		const std::vector<Vertex>& getVertices() const { return m_vertices; }
		const std::vector<vec3ui>& getIndices() const { return m_indices; }

		std::vector<Vertex>& getVertices() { return m_vertices; }
		std::vector<vec3ui>& getIndices() { return m_indices; }

		void computeMeshData(MeshData<FloatType>& meshData) const {

			meshData.clear();

			meshData.m_Vertices.resize(m_vertices.size());
			meshData.m_FaceIndicesVertices.resize(m_indices.size());
			if (m_bHasColors) {
				meshData.m_Colors.resize(m_vertices.size());
			}
			if (m_bHasNormals)	{
				meshData.m_Normals.resize(m_vertices.size());
			}
			if (m_bHasTexCoords) {
				meshData.m_TextureCoords.resize(m_vertices.size());
			}
			for (size_t i = 0; i < m_indices.size(); i++) {
				meshData.m_FaceIndicesVertices[i][0] = m_indices[i].x;
				meshData.m_FaceIndicesVertices[i][1] = m_indices[i].y;
				meshData.m_FaceIndicesVertices[i][2] = m_indices[i].z;
			}

			for (size_t i = 0; i < m_vertices.size(); i++) {
				meshData.m_Vertices[i] = m_vertices[i].position;
				if (m_bHasColors)		meshData.m_Colors[i] = m_vertices[i].color;
				if (m_bHasNormals)		meshData.m_Normals[i] = m_vertices[i].normal;
				if (m_bHasTexCoords)	meshData.m_TextureCoords[i] = m_vertices[i].texCoord;
			}
		}

		MeshData<FloatType> computeMeshData() const {
			MeshData<FloatType> meshData;
			computeMeshData(meshData);
			return meshData;
		}


		bool hasNormals() const {
			return m_bHasNormals;
		}
		bool hasColors() const {
			return m_bHasColors;
		}
		bool hasTexCoords() const {
			return m_bHasTexCoords;
		}
		void setHasColors(bool b) {
			m_bHasColors = b;
		}


		std::pair<BinaryGrid3, Matrix4x4<FloatType>> voxelize(FloatType voxelSize, const BoundingBox3<FloatType>& bounds = BoundingBox3<FloatType>(), bool solid = false) const {

			BoundingBox3<FloatType> bb;
			if (bounds.isInitialized()) {
				bb = bounds;
			} else {
                bb = computeBoundingBox();
				bb.scale((FloatType)1 + (FloatType)3.0*voxelSize);	//safety margin
			}

			Matrix4x4<FloatType> worldToVoxel = Matrix4x4<FloatType>::scale((FloatType)1/voxelSize) * Matrix4x4<FloatType>::translation(-bb.getMin());

            std::pair<BinaryGrid3, Matrix4x4<FloatType>> gridTrans = std::make_pair(BinaryGrid3(ml::math::max(vec3ui(bb.getExtent() / voxelSize), 1U)), worldToVoxel);

			voxelize(gridTrans.first, gridTrans.second, solid);

			return gridTrans;
		}


		void voxelize(BinaryGrid3& grid, const mat4f& worldToVoxel = mat4f::identity(), bool solid = false, bool verbose = true) const {
			for (size_t i = 0; i < m_indices.size(); i++) {
				vec3<FloatType> p0 = worldToVoxel * m_vertices[m_indices[i].x].position;
				vec3<FloatType> p1 = worldToVoxel * m_vertices[m_indices[i].y].position;
				vec3<FloatType> p2 = worldToVoxel * m_vertices[m_indices[i].z].position;

				BoundingBox3<FloatType> bb0(p0, p1, p2);
                //
                // TODO MATTHIAS: this + 1.0f should be investigated more.
                //
				BoundingBox3<FloatType> bb1(vec3<FloatType>(0,0,0), vec3<FloatType>((FloatType)grid.getDimX() + 1.0f, (FloatType)grid.getDimY() + 1.0f, (FloatType)grid.getDimZ() + 1.0f));
				if (bb0.intersects(bb1)) {
					voxelizeTriangle(p0, p1, p2, grid, solid);
				} else if (verbose) {
					std::cerr << "out of bounds: " << p0 << "\tof: " << grid.getDimensions() << std::endl;
					std::cerr << "out of bounds: " << p1 << "\tof: " << grid.getDimensions() << std::endl;
					std::cerr << "out of bounds: " << p2 << "\tof: " << grid.getDimensions() << std::endl;
					MLIB_WARNING("triangle outside of grid - ignored");
				}
			}
		}
	//private:

		void voxelizeTriangle(const vec3<FloatType>& v0, const vec3<FloatType>& v1, const vec3<FloatType>& v2, BinaryGrid3& grid, bool solid = false) const {

			FloatType diagLenSq = (FloatType)3.0;
			if ((v0-v1).lengthSq() < diagLenSq && (v0-v2).lengthSq() < diagLenSq &&	(v1-v2).lengthSq() < diagLenSq) {
				BoundingBox3<FloatType> bb(v0, v1, v2);
				vec3ui minI = math::floor(bb.getMin());
				vec3ui maxI = math::ceil(bb.getMax());
				minI = vec3ui(math::clamp(minI.x, 0u, (unsigned int)grid.getDimX()), math::clamp(minI.y, 0u, (unsigned int)grid.getDimY()), math::clamp(minI.z, 0u, (unsigned int)grid.getDimZ()));
				maxI = vec3ui(math::clamp(maxI.x, 0u, (unsigned int)grid.getDimX()), math::clamp(maxI.y, 0u, (unsigned int)grid.getDimY()), math::clamp(maxI.z, 0u, (unsigned int)grid.getDimZ()));

				//test for accurate voxel-triangle intersections
				for (unsigned int i = minI.x; i <= maxI.x; i++) {
					for (unsigned int j = minI.y; j <= maxI.y; j++) {
						for (unsigned int k = minI.z; k <= maxI.z; k++) {
							vec3<FloatType> v((FloatType)i,(FloatType)j,(FloatType)k);
							BoundingBox3<FloatType> voxel;
							const FloatType eps = (FloatType)0.0000;
							voxel.include((v - (FloatType)0.5-eps));
							voxel.include((v + (FloatType)0.5+eps));
							if (voxel.intersects(v0, v1, v2)) {
								if (solid) {
									//project to xy-plane
									vec2<FloatType> pv = v.getVec2();
                                    if (intersection::intersectTrianglePoint(v0.getVec2(), v1.getVec2(), v2.getVec2(), pv)) {
										Ray<FloatType> r0(vec3<FloatType>(v), vec3<FloatType>(0,0,1));
										Ray<FloatType> r1(vec3<FloatType>(v), vec3<FloatType>(0,0,-1));
										FloatType t0, t1, _u0, _u1, _v0, _v1;
										bool b0 = intersection::intersectRayTriangle(v0,v1,v2,r0,t0,_u0,_v0);
										bool b1 = intersection::intersectRayTriangle(v0,v1,v2,r1,t1,_u1,_v1);
										if ((b0 && t0 <= (FloatType)0.5) || (b1 && t1 <= (FloatType)0.5)) {
											if (i < grid.getDimX() && j < grid.getDimY() && k < grid.getDimZ()) {
												grid.toggleVoxelAndBehindSlice(i, j, k);
											}
										}
										//grid.setVoxel(i,j,k);
									}
								} else {
									if (i < grid.getDimX() && j < grid.getDimY() && k < grid.getDimZ()) {
										grid.setVoxel(i, j, k);
									}
								}
							}
						}
					}
				} 
			} else {
				vec3<FloatType> e0 = (FloatType)0.5*(v0 + v1);
				vec3<FloatType> e1 = (FloatType)0.5*(v1 + v2);
				vec3<FloatType> e2 = (FloatType)0.5*(v2 + v0);
				voxelizeTriangle(v0,e0,e2, grid, solid);
				voxelizeTriangle(e0,v1,e1, grid, solid);
				voxelizeTriangle(e1,v2,e2, grid, solid);
				voxelizeTriangle(e0,e1,e2, grid, solid);
			}
		}

    // boost archive serialization
    friend class boost::serialization::access;
    template<class Archive>
    inline void serialize(Archive& ar, const unsigned int version) {
      ar & m_vertices & m_indices;
      if (version >= 1) {
        ar & m_bHasColors & m_bHasNormals & m_bHasTexCoords;
      }
    }

		bool m_bHasNormals;
		bool m_bHasTexCoords;
		bool m_bHasColors;

		std::vector<Vertex>		m_vertices;
		std::vector<vec3ui>		m_indices;
	};

	typedef TriMesh<float> TriMeshf;
	typedef TriMesh<double> TriMeshd;

    //
    // Matthias TODO: I made all these things public because I don't know how private serialization works on BinaryDataBuffer stuff!
    //
    template<class BinaryDataBuffer, class BinaryDataCompressor, class FloatType>
    inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<< (BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const TriMesh<FloatType> &m) {
        s << m.m_bHasNormals << m.m_bHasTexCoords << m.m_bHasColors;
        s.writePrimitive(m.m_vertices);
        s.writePrimitive(m.m_indices);
        return s;
    }

    template<class BinaryDataBuffer, class BinaryDataCompressor, class FloatType>
    inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>> (BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, TriMesh<FloatType> &m) {
        s >> m.m_bHasNormals >> m.m_bHasTexCoords >> m.m_bHasColors;
        s.readPrimitive(m.m_vertices);
        s.readPrimitive(m.m_indices);
        return s;
    }


	template<class FloatType>
	std::ostream& operator<<(std::ostream& os, const TriMesh<FloatType>& triMesh) {
		os << "TriMesh:\n"
			<< "\tVertices:  " << triMesh.m_vertices.size() << "\n"
			<< "\tIndices:   " << triMesh.m_indices.size() << "*3\n"
			<< std::endl;

		return os;
	}


}  // namespace ml

#include "triMesh.cpp"

#endif  // INCLUDE_CORE_MESH_TRIMESH_H_
