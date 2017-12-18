#ifndef CORE_MESH_MESHSHAPES_H_
#define CORE_MESH_MESHSHAPES_H_

namespace ml {

template<class FloatType>
class Shapes {
public:

	static TriMesh<FloatType> rectangleZ(const vec2<FloatType> &start, const vec2<FloatType> &end, FloatType zValue = FloatType(), const vec4<FloatType>& color = ml::vec4<FloatType>(1, 1, 1, 1));

	static TriMesh<FloatType> quad(const vec3<FloatType> &p0, const vec3<FloatType> &p1, const vec3<FloatType> &p2, const vec3<FloatType> &p3, const vec4<FloatType>& color = ml::vec4<FloatType>(1, 1, 1, 1));

    static TriMesh<FloatType> box(FloatType xDim, FloatType yDim, FloatType zDim, const vec4<FloatType>& color = ml::vec4<FloatType>(1, 1, 1, 1));

    static TriMesh<FloatType> box(const OrientedBoundingBox3f &obb, const vec4<FloatType>& color = ml::vec4<FloatType>(1, 1, 1, 1));

    static TriMesh<FloatType> box(const BoundingBox3<FloatType> &bbox, const vec4<FloatType>& color = vec4<FloatType>(1, 1, 1, 1));

    static TriMesh<FloatType> box(FloatType dim = 1, const vec4<FloatType>& color = vec4<FloatType>(1, 1, 1, 1)) {
        return box(dim, dim, dim, color);
    }

    static TriMesh<FloatType> cylinder(FloatType radius, FloatType height, UINT stacks, UINT slices, const vec4<FloatType>& color = ml::vec4<FloatType>(1, 1, 1, 1));

    static TriMesh<FloatType> cylinder(const vec3<FloatType> &p0, const vec3<FloatType> &p1, FloatType radius, UINT stacks, UINT slices, const vec4<FloatType>& color = ml::vec4<FloatType>(1, 1, 1, 1));

    static TriMesh<FloatType> torus(const vec3<FloatType> &center, FloatType majorRadius, FloatType minorRadius, UINT stacks, UINT slices, const vec4<FloatType>& color = ml::vec4<FloatType>(1, 1, 1, 1));

    static TriMesh<FloatType> torus(const vec3<FloatType> &center, FloatType majorRadius, FloatType minorRadius, UINT stacks, UINT slices, const std::function<vec4<FloatType>(unsigned int)> &stackIndexToColor);

    static TriMesh<FloatType> line(const vec3<FloatType>& p0, const vec3<FloatType>& p1, const vec4<FloatType>& color, const FloatType thickness) {
      return cylinder(p0, p1, thickness, 2, 9, color);
    }

    static TriMesh<FloatType> wireframeBox(FloatType dimension, const vec4<FloatType>& color, FloatType thickness);

    static TriMesh<FloatType> wireframeBox(const mat4f& unitCubeToWorld, const vec4<FloatType>& color, FloatType thickness);

    static TriMesh<FloatType> sphere(const FloatType radius, const ml::vec3<FloatType>& pos, const size_t stacks = 10, const size_t slices = 10, const ml::vec4<FloatType>& color = ml::vec4<FloatType>(1, 1, 1, 1));

    static MeshData<FloatType> toMeshData(const BoundingBox3<FloatType>& s, const vec4<FloatType>& color = vec4<FloatType>(1, 1, 1, 1), bool bottomPlaneOnly = false) {
	    MeshData<FloatType> meshData;	std::vector<vec3ui> indices;
	    if (bottomPlaneOnly) {
		    s.makeTriMeshBottomPlane(meshData.m_Vertices, indices, meshData.m_Normals);
	    } else {
		    s.makeTriMesh(meshData.m_Vertices, indices, meshData.m_Normals);
	    }
	    //meshData.m_FaceIndicesVertices.resize(indices.size(), std::vector<unsigned int>(3));
	    meshData.m_FaceIndicesVertices.resize(indices.size(), 3);
	    for (size_t i = 0; i < indices.size(); i++) {
		    meshData.m_FaceIndicesVertices[i][0] = indices[i].x;
		    meshData.m_FaceIndicesVertices[i][1] = indices[i].y;
		    meshData.m_FaceIndicesVertices[i][2] = indices[i].z;
	    }
	    meshData.m_Colors.resize(meshData.m_Vertices.size(), color);
	    return meshData;
    }

    static TriMesh<FloatType> unifyMeshes(const std::vector<TriMesh<FloatType> > &meshes)
    {
        std::vector<vec3<FloatType> > unifiedVertices;
        std::vector<unsigned int> unifiedIndices;
        std::vector<vec4<FloatType> > unifiedColors;
        std::vector<vec2<FloatType> > unifiedTexCoords;
        std::vector<vec3<FloatType> > unifiedNormals;

        int meshBaseVertex = 0;
        for (const auto &mesh : meshes)
        {
            for (auto &v : mesh.getVertices())
            {
                unifiedVertices.push_back(v.position);
                unifiedColors.push_back(v.color);
            }
            for (auto &i : mesh.getIndices())
            {
                unifiedIndices.push_back(i.x + meshBaseVertex);
                unifiedIndices.push_back(i.y + meshBaseVertex);
                unifiedIndices.push_back(i.z + meshBaseVertex);
            }
            meshBaseVertex += (int)mesh.getVertices().size();
        }

        return TriMesh<FloatType>(unifiedVertices, unifiedIndices, unifiedColors, unifiedNormals, unifiedTexCoords);
    }
};

typedef Shapes<float> Shapesf;
typedef Shapes<double> Shapesd;

}  // namespace ml

#include "meshShapes.cpp"

#endif  // INCLUDE_CORE_MESH_MESHSHAPES_H_
