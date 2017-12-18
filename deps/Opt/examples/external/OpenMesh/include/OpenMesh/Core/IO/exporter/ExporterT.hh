/* ========================================================================= *
 *                                                                           *
 *                               OpenMesh                                    *
 *           Copyright (c) 2001-2015, RWTH-Aachen University                 *
 *           Department of Computer Graphics and Multimedia                  *
 *                          All rights reserved.                             *
 *                            www.openmesh.org                               *
 *                                                                           *
 *---------------------------------------------------------------------------*
 * This file is part of OpenMesh.                                            *
 *---------------------------------------------------------------------------*
 *                                                                           *
 * Redistribution and use in source and binary forms, with or without        *
 * modification, are permitted provided that the following conditions        *
 * are met:                                                                  *
 *                                                                           *
 * 1. Redistributions of source code must retain the above copyright notice, *
 *    this list of conditions and the following disclaimer.                  *
 *                                                                           *
 * 2. Redistributions in binary form must reproduce the above copyright      *
 *    notice, this list of conditions and the following disclaimer in the    *
 *    documentation and/or other materials provided with the distribution.   *
 *                                                                           *
 * 3. Neither the name of the copyright holder nor the names of its          *
 *    contributors may be used to endorse or promote products derived from   *
 *    this software without specific prior written permission.               *
 *                                                                           *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED *
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A           *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       *
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        *
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              *
 *                                                                           *
 * ========================================================================= */

/*===========================================================================*\
 *                                                                           *
 *   $Revision$                                                         *
 *   $Date$                   *
 *                                                                           *
\*===========================================================================*/


//=============================================================================
//
//  Implements an exporter module for arbitrary OpenMesh meshes
//
//=============================================================================


#ifndef __EXPORTERT_HH__
#define __EXPORTERT_HH__


//=== INCLUDES ================================================================

// C++
#include <vector>

// OpenMesh
#include <OpenMesh/Core/System/config.h>
#include <OpenMesh/Core/Geometry/VectorT.hh>
#include <OpenMesh/Core/Utils/GenProg.hh>
#include <OpenMesh/Core/Utils/vector_cast.hh>
#include <OpenMesh/Core/Utils/color_cast.hh>
#include <OpenMesh/Core/IO/exporter/BaseExporter.hh>


//=== NAMESPACES ==============================================================

namespace OpenMesh {
namespace IO {


//=== EXPORTER CLASS ==========================================================

/**
 *  This class template provides an exporter module for OpenMesh meshes.
 */
template <class Mesh>
class ExporterT : public BaseExporter
{
public:

  // Constructor
  ExporterT(const Mesh& _mesh) : mesh_(_mesh) {}


  // get vertex data

  Vec3f  point(VertexHandle _vh)    const
  {
    return vector_cast<Vec3f>(mesh_.point(_vh));
  }

  Vec3f  normal(VertexHandle _vh)   const
  {
    return (mesh_.has_vertex_normals()
	    ? vector_cast<Vec3f>(mesh_.normal(_vh))
	    : Vec3f(0.0f, 0.0f, 0.0f));
  }

  Vec3uc color(VertexHandle _vh)    const
  {
    return (mesh_.has_vertex_colors()
	    ? color_cast<Vec3uc>(mesh_.color(_vh))
	    : Vec3uc(0, 0, 0));
  }

  Vec4uc colorA(VertexHandle _vh)   const
  {
    return (mesh_.has_vertex_colors()
      ? color_cast<Vec4uc>(mesh_.color(_vh))
      : Vec4uc(0, 0, 0, 0));
  }

  Vec3ui colori(VertexHandle _vh)    const
  {
    return (mesh_.has_vertex_colors()
	    ? color_cast<Vec3ui>(mesh_.color(_vh))
	    : Vec3ui(0, 0, 0));
  }

  Vec4ui colorAi(VertexHandle _vh)   const
  {
    return (mesh_.has_vertex_colors()
      ? color_cast<Vec4ui>(mesh_.color(_vh))
      : Vec4ui(0, 0, 0, 0));
  }

  Vec3f colorf(VertexHandle _vh)    const
  {
    return (mesh_.has_vertex_colors()
	    ? color_cast<Vec3f>(mesh_.color(_vh))
	    : Vec3f(0, 0, 0));
  }

  Vec4f colorAf(VertexHandle _vh)   const
  {
    return (mesh_.has_vertex_colors()
      ? color_cast<Vec4f>(mesh_.color(_vh))
      : Vec4f(0, 0, 0, 0));
  }

  Vec2f  texcoord(VertexHandle _vh) const
  {
#if defined(OM_CC_GCC) && (OM_CC_VERSION<30000)
    // Workaround!
    // gcc 2.95.3 exits with internal compiler error at the
    // code below!??? **)
    if (mesh_.has_vertex_texcoords2D())
      return vector_cast<Vec2f>(mesh_.texcoord2D(_vh));
    return Vec2f(0.0f, 0.0f);
#else // **)
    return (mesh_.has_vertex_texcoords2D()
	    ? vector_cast<Vec2f>(mesh_.texcoord2D(_vh))
	    : Vec2f(0.0f, 0.0f));
#endif
  }

  // get edge data

  Vec3uc color(EdgeHandle _eh)    const
  {
      return (mesh_.has_edge_colors()
      ? color_cast<Vec3uc>(mesh_.color(_eh))
      : Vec3uc(0, 0, 0));
  }

  Vec4uc colorA(EdgeHandle _eh)   const
  {
      return (mesh_.has_edge_colors()
      ? color_cast<Vec4uc>(mesh_.color(_eh))
      : Vec4uc(0, 0, 0, 0));
  }

  Vec3ui colori(EdgeHandle _eh)    const
  {
      return (mesh_.has_edge_colors()
      ? color_cast<Vec3ui>(mesh_.color(_eh))
      : Vec3ui(0, 0, 0));
  }

  Vec4ui colorAi(EdgeHandle _eh)   const
  {
      return (mesh_.has_edge_colors()
      ? color_cast<Vec4ui>(mesh_.color(_eh))
      : Vec4ui(0, 0, 0, 0));
  }

  Vec3f colorf(EdgeHandle _eh)    const
  {
    return (mesh_.has_vertex_colors()
	    ? color_cast<Vec3f>(mesh_.color(_eh))
	    : Vec3f(0, 0, 0));
  }

  Vec4f colorAf(EdgeHandle _eh)   const
  {
    return (mesh_.has_vertex_colors()
      ? color_cast<Vec4f>(mesh_.color(_eh))
      : Vec4f(0, 0, 0, 0));
  }

  // get face data

  unsigned int get_vhandles(FaceHandle _fh,
			    std::vector<VertexHandle>& _vhandles) const
  {
    unsigned int count(0);
    _vhandles.clear();
    for (typename Mesh::CFVIter fv_it=mesh_.cfv_iter(_fh); fv_it.is_valid(); ++fv_it)
    {
      _vhandles.push_back(*fv_it);
      ++count;
    }
    return count;
  }

  Vec3f  normal(FaceHandle _fh)   const
  {
    return (mesh_.has_face_normals()
            ? vector_cast<Vec3f>(mesh_.normal(_fh))
            : Vec3f(0.0f, 0.0f, 0.0f));
  }

  Vec3uc  color(FaceHandle _fh)   const
  {
    return (mesh_.has_face_colors()
            ? color_cast<Vec3uc>(mesh_.color(_fh))
            : Vec3uc(0, 0, 0));
  }

  Vec4uc  colorA(FaceHandle _fh)   const
  {
    return (mesh_.has_face_colors()
            ? color_cast<Vec4uc>(mesh_.color(_fh))
            : Vec4uc(0, 0, 0, 0));
  }

  Vec3ui  colori(FaceHandle _fh)   const
  {
    return (mesh_.has_face_colors()
            ? color_cast<Vec3ui>(mesh_.color(_fh))
            : Vec3ui(0, 0, 0));
  }

  Vec4ui  colorAi(FaceHandle _fh)   const
  {
    return (mesh_.has_face_colors()
            ? color_cast<Vec4ui>(mesh_.color(_fh))
            : Vec4ui(0, 0, 0, 0));
  }

  Vec3f colorf(FaceHandle _fh)    const
  {
    return (mesh_.has_vertex_colors()
	    ? color_cast<Vec3f>(mesh_.color(_fh))
	    : Vec3f(0, 0, 0));
  }

  Vec4f colorAf(FaceHandle _fh)   const
  {
    return (mesh_.has_vertex_colors()
      ? color_cast<Vec4f>(mesh_.color(_fh))
      : Vec4f(0, 0, 0, 0));
  }

  virtual const BaseKernel* kernel() { return &mesh_; }


  // query number of faces, vertices, normals, texcoords
  size_t n_vertices()  const { return mesh_.n_vertices(); }
  size_t n_faces()     const { return mesh_.n_faces(); }
  size_t n_edges()     const { return mesh_.n_edges(); }


  // property information
  bool is_triangle_mesh() const
  { return Mesh::is_triangles(); }

  bool has_vertex_normals()   const { return mesh_.has_vertex_normals();   }
  bool has_vertex_colors()    const { return mesh_.has_vertex_colors();    }
  bool has_vertex_texcoords() const { return mesh_.has_vertex_texcoords2D(); }
  bool has_edge_colors()      const { return mesh_.has_edge_colors();      }
  bool has_face_normals()     const { return mesh_.has_face_normals();     }
  bool has_face_colors()      const { return mesh_.has_face_colors();      }

private:

   const Mesh& mesh_;
};


//=============================================================================
} // namespace IO
} // namespace OpenMesh
//=============================================================================
#endif
//=============================================================================
