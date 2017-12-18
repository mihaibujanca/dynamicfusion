namespace ml {

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::rectangleZ(const vec2<FloatType> &start, const vec2<FloatType> &end, FloatType zValue, const vec4<FloatType>& color)
	{
		std::vector<vec3<FloatType>> vertices(4);
		std::vector<UINT> indices = { 0, 1, 2, 0, 2, 3 };
		std::vector<vec3<FloatType>> normals(4, vec3f::eZ);
		std::vector<vec2<FloatType>> texCoords(4);
		std::vector<vec4<FloatType>> colors = { color, color, color, color };

		vertices[0] = vec3<FloatType>(start.x, start.y, zValue);
		vertices[1] = vec3<FloatType>(end.x, start.y, zValue);
		vertices[2] = vec3<FloatType>(end.x, end.y, zValue);
		vertices[3] = vec3<FloatType>(start.x, end.y, zValue);

		texCoords[0] = vec2<FloatType>(0.0f, 0.0f);
		texCoords[1] = vec2<FloatType>(1.0f, 0.0f);
		texCoords[2] = vec2<FloatType>(1.0f, 1.0f);
		texCoords[3] = vec2<FloatType>(0.0f, 1.0f);

		return TriMesh<FloatType>(vertices.size(), indices.size(), vertices.data(), indices.data(),
			colors.data(), normals.data(), texCoords.data());
	}

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::quad(const vec3<FloatType> &p0, const vec3<FloatType> &p1, const vec3<FloatType> &p2, const vec3<FloatType> &p3, const vec4<FloatType>& color)
	{
		std::vector<vec3<FloatType>> vertices = { p0, p1, p2, p3 };
		std::vector<UINT> indices = { 0, 1, 2, 0, 2, 3 };
		std::vector<vec3<FloatType>> normals = { ((p1 - p0) ^ (p3 - p0)).getNormalized(),
			((p0 - p1) ^ (p2 - p1)).getNormalized(),
			((p1 - p2) ^ (p3 - p2)).getNormalized(),
			((p0 - p3) ^ (p2 - p3)).getNormalized() };
		std::vector<vec2<FloatType>> texCoords(4);
		std::vector<vec4<FloatType>> colors = { color, color, color, color };

		texCoords[0] = vec2<FloatType>(1.0f, 1.0f);
		texCoords[1] = vec2<FloatType>(0.0f, 1.0f);
		texCoords[2] = vec2<FloatType>(0.0f, 0.0f);
		texCoords[3] = vec2<FloatType>(1.0f, 0.0f);

		return TriMesh<FloatType>(vertices.size(), indices.size(), vertices.data(), indices.data(),
			colors.data(), normals.data(), texCoords.data());
	}

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::box(const BoundingBox3<FloatType> &bbox, const vec4<FloatType>& color)
	{
		auto extent = bbox.getExtent();
		auto center = bbox.getCenter();
		TriMesh<FloatType> result = box(extent.x, extent.y, extent.z, color);
		for (auto &v : result.getVertices())
		{
			v.position += center;
		}
		return result;
	}

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::box(const OrientedBoundingBox3f &obb, const vec4<FloatType>& color)
	{
		TriMesh<FloatType> result = box(BoundingBox3<FloatType>(vec3<FloatType>::origin, vec3<FloatType>(1.0f, 1.0f, 1.0f)), color);

		result.transform(obb.getOBBToWorld());

		return result;
	}

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::box(FloatType xDim, FloatType yDim, FloatType zDim, const vec4<FloatType>& color) {

		FloatType cubeVData[8][3] = {
			{ 1.0f, 1.0f, 1.0f }, { -1.0f, 1.0f, 1.0f }, { -1.0f, -1.0f, 1.0f },
			{ 1.0f, -1.0f, 1.0f }, { 1.0f, 1.0f, -1.0f }, { -1.0f, 1.0f, -1.0f },
			{ -1.0f, -1.0f, -1.0f }, { 1.0f, -1.0f, -1.0f }
		};

		int cubeIData[12][3] = {
			{ 1, 2, 3 }, { 1, 3, 0 }, { 0, 3, 7 }, { 0, 7, 4 }, { 3, 2, 6 },
			{ 3, 6, 7 }, { 1, 6, 2 }, { 1, 5, 6 }, { 0, 5, 1 }, { 0, 4, 5 },
			{ 6, 5, 4 }, { 6, 4, 7 }
		};

		int cubeEData[12][2] = {
			{ 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
			{ 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
			{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
		};

		typename std::vector<typename TriMesh<FloatType>::Vertex> vv(8);
		std::vector<UINT> vi(12 * 3);

		// Vertices
		for (int i = 0; i < 8; i++) {
			vv[i].position = vec3<FloatType>(cubeVData[i][0], cubeVData[i][1], cubeVData[i][2]);
			vv[i].normal = vec3<FloatType>(1.0f, 0.0f, 0.0f);  // TODO(ms): write and call generateNormals() function
			vv[i].color = color;
		}

		// Triangles
		for (int i = 0; i < 12; i++) {
			vi[i * 3 + 0] = cubeIData[i][0];
			vi[i * 3 + 1] = cubeIData[i][1];
			vi[i * 3 + 2] = cubeIData[i][2];
		}

		TriMesh<FloatType> mesh(vv, vi);
		mesh.scale(vec3<FloatType>(0.5f * xDim, 0.5f * yDim, 0.5f * zDim));
		mesh.setHasColors(true);

		return mesh;
	}

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::cylinder(FloatType radius, FloatType height, UINT stacks, UINT slices, const vec4<FloatType>& color) {
		std::vector<typename TriMesh<FloatType>::Vertex> vertices((stacks + 1) * slices);
		std::vector<UINT> indices(stacks * slices * 6);

		UINT vIndex = 0;
		for (UINT i = 0; i <= stacks; i++)
			for (UINT i2 = 0; i2 < slices; i2++)
			{
				auto& vtx = vertices[vIndex++];
				FloatType theta = FloatType(i2) * 2.0f * math::PIf / FloatType(slices);
				vtx.position = vec3<FloatType>(radius * cosf(theta), radius * sinf(theta), height * FloatType(i) / FloatType(stacks));
				vtx.normal = vec3<FloatType>(1.0f, 0.0f, 0.0f);  // TODO(ms): write and call generateNormals() function
				vtx.color = color;
			}

		UINT iIndex = 0;
		for (UINT i = 0; i < stacks; i++)
			for (UINT i2 = 0; i2 < slices; i2++)
			{
				int i2p1 = (i2 + 1) % slices;

				indices[iIndex++] = (i + 1) * slices + i2;
				indices[iIndex++] = i * slices + i2;
				indices[iIndex++] = i * slices + i2p1;


				indices[iIndex++] = (i + 1) * slices + i2;
				indices[iIndex++] = i * slices + i2p1;
				indices[iIndex++] = (i + 1) * slices + i2p1;
			}

		return TriMesh<FloatType>(vertices, indices, true, false, false, true); // has colors
	}

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::cylinder(const vec3<FloatType>& p0, const vec3<FloatType>& p1, FloatType radius, UINT stacks, UINT slices, const vec4<FloatType>& color) {
		FloatType height = (p1 - p0).length();

		TriMesh<FloatType> result = Shapesf::cylinder(radius, height, stacks, slices, color);
		result.transform(mat4f::translation(p0) * mat4f::face(vec3<FloatType>::eZ, p1 - p0));
		return result;
	}

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::torus(const vec3<FloatType> &center, FloatType majorRadius, FloatType minorRadius, UINT stacks, UINT slices, const vec4<FloatType>& color)
	{
		return torus(center, majorRadius, minorRadius, stacks, slices, [&](unsigned int stackIndex) { return color; });
	}

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::torus(const vec3<FloatType> &center, FloatType majorRadius, FloatType minorRadius, UINT stacks, UINT slices, const std::function<vec4<FloatType>(unsigned int)> &stackIndexToColor)
	{
		std::vector<typename TriMesh<FloatType>::Vertex> vertices(slices * stacks);
		std::vector<UINT> indices(stacks * slices * 6);

		UINT vIndex = 0;
		// initial theta faces y front
		FloatType baseTheta = ml::math::PIf / 2.0f;
		for (UINT i = 0; i < stacks; i++)
		{
			FloatType theta = FloatType(i) * 2.0f * ml::math::PIf / FloatType(stacks) + baseTheta;
			auto color = stackIndexToColor(i);
			FloatType sinT = sinf(theta);
			FloatType cosT = cosf(theta);
			ml::vec3<FloatType> t0(cosT * majorRadius, sinT * majorRadius, 0.0f);
			for (UINT i2 = 0; i2 < slices; i2++)
			{
				auto& vtx = vertices[vIndex++];

				FloatType phi = FloatType(i2) * 2.0f * ml::math::PIf / FloatType(slices);
				FloatType sinP = sinf(phi);
				vtx.position = ml::vec3<FloatType>(minorRadius * cosT * sinP, minorRadius * sinT * sinP, minorRadius * cosf(phi)) + t0;
				vtx.color = color;
			}
		}

		UINT iIndex = 0;
		for (UINT i = 0; i < stacks; i++)
		{
			UINT ip1 = (i + 1) % stacks;
			for (UINT i2 = 0; i2 < slices; i2++)
			{
				UINT i2p1 = (i2 + 1) % slices;

				indices[iIndex++] = ip1 * slices + i2;
				indices[iIndex++] = i * slices + i2;
				indices[iIndex++] = i * slices + i2p1;

				indices[iIndex++] = ip1 * slices + i2;
				indices[iIndex++] = i * slices + i2p1;
				indices[iIndex++] = ip1 * slices + i2p1;
			}
		}

		return TriMesh<FloatType>(vertices, indices, true);
	}

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::wireframeBox(FloatType dim, const vec4<FloatType>& color, FloatType thickness) {
		MLIB_WARNING("untested function");
		FloatType cubeVData[8][3] = {
			{ 1.0f, 1.0f, 1.0f }, { -1.0f, 1.0f, 1.0f }, { -1.0f, -1.0f, 1.0f },
			{ 1.0f, -1.0f, 1.0f }, { 1.0f, 1.0f, -1.0f }, { -1.0f, 1.0f, -1.0f },
			{ -1.0f, -1.0f, -1.0f }, { 1.0f, -1.0f, -1.0f }
		};

		int cubeEData[12][2] = {
			{ 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
			{ 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
			{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
		};

		std::vector<ml::TriMesh<FloatType>> meshes;
		ml::vec3<FloatType> v[8];  std::memmove(v, cubeVData, sizeof(v[0]) * 8);
		for (uint i = 0; i < 12; i++) {
			meshes.push_back(line(dim * v[cubeEData[i][0]], dim * v[cubeEData[i][1]], color, thickness));
		}
		return meshutil::createUnifiedMesh(meshes);
	}

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::wireframeBox(const mat4f& xf, const vec4<FloatType>& color, FloatType thickness) {

		FloatType cubeVData[8][3] = {
			{ 1.0f, 1.0f, 1.0f }, { -1.0f, 1.0f, 1.0f }, { -1.0f, -1.0f, 1.0f },
			{ 1.0f, -1.0f, 1.0f }, { 1.0f, 1.0f, -1.0f }, { -1.0f, 1.0f, -1.0f },
			{ -1.0f, -1.0f, -1.0f }, { 1.0f, -1.0f, -1.0f }
		};

		int cubeEData[12][2] = {
			{ 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
			{ 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
			{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
		};


		std::vector<ml::TriMesh<FloatType>> meshes;
		ml::vec3<FloatType> v[8];  std::memmove(v, cubeVData, sizeof(v[0]) * 8);
		for (uint i = 0; i < 8; i++) { v[i] = xf * v[i]; }
		for (unsigned int i = 0; i < 12; i++) {
			const ml::vec3<FloatType>& p0 = v[cubeEData[i][0]];
			const ml::vec3<FloatType>& p1 = v[cubeEData[i][1]];
			meshes.push_back(line(p0, p1, color, thickness));
		}
		return meshutil::createUnifiedMesh(meshes);
	}

	template<class FloatType>
	TriMesh<FloatType> Shapes<FloatType>::sphere(const FloatType radius, const ml::vec3<FloatType>& pos, const size_t stacks /*= 10*/, const size_t slices /*= 10*/, const ml::vec4<FloatType>& color /*= ml::vec4<FloatType>(1,1,1,1) */) {
		MeshDataf meshdata;
		auto& V = meshdata.m_Vertices;
		auto& I = meshdata.m_FaceIndicesVertices;
		auto& N = meshdata.m_Normals;
		auto& C = meshdata.m_Colors;
		const FloatType thetaDivisor = 1.0f / stacks * ml::math::PIf;
		const FloatType phiDivisor = 1.0f / slices * 2.0f * ml::math::PIf;
		for (size_t t = 0; t < stacks; t++) { // stacks increment elevation (theta)
			FloatType theta1 = t * thetaDivisor;
			FloatType theta2 = (t + 1) * thetaDivisor;

			for (size_t p = 0; p < slices; p++) { // slices increment azimuth (phi)
				FloatType phi1 = p * phiDivisor;
				FloatType phi2 = (p + 1) * phiDivisor;

				const auto sph2xyz = [&](FloatType r, FloatType theta, FloatType phi) {
					const FloatType sinTheta = sinf(theta), sinPhi = sinf(phi), cosTheta = cosf(theta), cosPhi = cosf(phi);
					return ml::vec3<FloatType>(r * sinTheta * cosPhi, r * sinTheta * sinPhi, r * cosTheta);
				};

				// phi2   phi1
				//  |      |
				//  2------1 -- theta1
				//  |\ _   |
				//  |    \ |
				//  3------4 -- theta2
				//  
				// Points
				const ml::vec3<FloatType>
					r1 = sph2xyz(radius, theta1, phi1),
					r2 = sph2xyz(radius, theta1, phi2),
					r3 = sph2xyz(radius, theta2, phi2),
					r4 = sph2xyz(radius, theta2, phi1);
				V.push_back(r1 + pos);
				V.push_back(r2 + pos);
				V.push_back(r3 + pos);
				V.push_back(r4 + pos);

				// Colors
				for (int i = 0; i < 4; i++) {
					C.push_back(color);
				}

				// Normals
				N.push_back(r1.getNormalized());
				N.push_back(r2.getNormalized());
				N.push_back(r3.getNormalized());
				N.push_back(r4.getNormalized());

				const UINT baseIdx = static_cast<UINT>(t * slices * 4 + p * 4);

				// Indices
				std::vector<unsigned int> indices;
				if (t == 0) {  // top cap -- t1p1, t2p2, t2p1
					indices.push_back(baseIdx + 0);
					indices.push_back(baseIdx + 3);
					indices.push_back(baseIdx + 2);
					I.push_back(indices);
				}
				else if (t + 1 == stacks) {  // bottom cap -- t2p2, t1p1, t1p2
					indices.push_back(baseIdx + 2);
					indices.push_back(baseIdx + 1);
					indices.push_back(baseIdx + 0);
					I.push_back(indices);
				}
				else {  // regular piece
					indices.push_back(baseIdx + 0);
					indices.push_back(baseIdx + 3);
					indices.push_back(baseIdx + 1);
					I.push_back(indices);
					indices.clear();
					indices.push_back(baseIdx + 1);
					indices.push_back(baseIdx + 3);
					indices.push_back(baseIdx + 2);
					I.push_back(indices);
				}
			}
		}
		meshdata.mergeCloseVertices(0.00001f, true);
		return TriMesh<FloatType>(meshdata);
	}

}  // namespace ml