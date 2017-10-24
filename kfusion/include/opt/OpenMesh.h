#pragma once

// Includes OpenMesh
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define _USE_MATH_DEFINES

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Utils/Property.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>

using namespace OpenMesh;

struct Traits : DefaultTraits
{
	typedef OpenMesh::Vec3f Point;
	typedef OpenMesh::Vec3f Normal; 
	typedef OpenMesh::Vec3uc Color;
	typedef float Scalar;

	VertexTraits
	{
		public:

			VertexT() : m_constrained(false)
			{
			}

			bool m_constrained;
	};

	VertexAttributes(Attributes::Status| Attributes::Normal | Attributes::Color);
	FaceAttributes(Attributes::Status);
	EdgeAttributes(Attributes::Status);
};

typedef TriMesh_ArrayKernelT<Traits> SimpleMesh;
