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
 *   $Revision$                                                              *
 *   $Date$                                                                  *
 *                                                                           *
\*===========================================================================*/

#ifndef PROPERTYMANAGER_HH_
#define PROPERTYMANAGER_HH_

#include <sstream>
#include <stdexcept>
#include <string>

namespace OpenMesh {

/**
 * This class is intended to manage the lifecycle of properties.
 * It also defines convenience operators to access the encapsulated
 * property's value.
 *
 * Usage example:
 *
 * \code
 * TriMesh mesh;
 * PropertyManager<VPropHandleT<bool>, TriMesh> visited(mesh, "visited.plugin-example.i8.informatik.rwth-aachen.de");
 *
 * for (TriMesh::VertexIter vh_it = mesh.begin(); ... ; ...) {
 *     if (!visited[*vh_it]) {
 *         visitComponent(mesh, *vh_it, visited);
 *     }
 * }
 * \endcode
 *
 */
template<typename PROPTYPE, typename MeshT>
class PropertyManager {
#if __cplusplus > 199711L || defined(__GXX_EXPERIMENTAL_CXX0X__)
    public:
        PropertyManager(const PropertyManager&) = delete;
        PropertyManager& operator=(const PropertyManager&) = delete;
#else
    private:
        /**
         * Noncopyable because there aren't no straightforward copy semantics.
         */
        PropertyManager(const PropertyManager&);

        /**
         * Noncopyable because there aren't no straightforward copy semantics.
         */
        PropertyManager& operator=(const PropertyManager&);
#endif

    public:
        /**
         * Constructor.
         *
         * Throws an \p std::runtime_error if \p existing is true and
         * no property named \p propname of the appropriate property type
         * exists.
         *
         * @param mesh The mesh on which to create the property.
         * @param propname The name of the property.
         * @param existing If false, a new property is created and its lifecycle is managed (i.e.
         * the property is deleted upon destruction of the PropertyManager instance). If true,
         * the instance merely acts as a convenience wrapper around an existing property with no
         * lifecycle management whatsoever.
         */
        PropertyManager(MeshT &mesh, const char *propname, bool existing = false) : mesh_(&mesh), retain_(existing), name_(propname) {
            if (existing) {
                if (!mesh_->get_property_handle(prop_, propname)) {
                    std::ostringstream oss;
                    oss << "Requested property handle \"" << propname << "\" does not exist.";
                    throw std::runtime_error(oss.str());
                }
            } else {
                mesh_->add_property(prop_, propname);
            }
        }

        PropertyManager() : mesh_(0), retain_(false) {
        }

        ~PropertyManager() {
            deleteProperty();
        }

        void swap(PropertyManager &rhs) {
            std::swap(mesh_, rhs.mesh_);
            std::swap(prop_, rhs.prop_);
            std::swap(retain_, rhs.retain_);
            std::swap(name_, rhs.name_);
        }

        static bool propertyExists(MeshT &mesh, const char *propname) {
            PROPTYPE dummy;
            return mesh.get_property_handle(dummy, propname);
        }

        bool isValid() const { return mesh_ != 0; }
        operator bool() const { return isValid(); }

        const PROPTYPE &getRawProperty() const { return prop_; }

        const std::string &getName() const { return name_; }

        MeshT &getMesh() const { return *mesh_; }

#if __cplusplus > 199711L || defined(__GXX_EXPERIMENTAL_CXX0X__)
        /// Only for pre C++11 compatibility.

        typedef PropertyManager<PROPTYPE, MeshT> Proxy;

        /**
         * Move constructor. Transfers ownership (delete responsibility).
         */
        PropertyManager(PropertyManager &&rhs) : mesh_(rhs.mesh_), prop_(rhs.prop_), retain_(rhs.retain_), name_(rhs.name_) {
            rhs.retain_ = true;
        }

        /**
         * Move assignment. Transfers ownership (delete responsibility).
         */
        PropertyManager &operator=(PropertyManager &&rhs) {

            deleteProperty();

            mesh_ = rhs.mesh_;
            prop_ = rhs.prop_;
            retain_ = rhs.retain_;
            name_ = rhs.name_;
            rhs.retain_ = true;

            return *this;
        }

        /**
         * Create a property manager for the supplied property and mesh.
         * If the property doesn't exist, it is created. In any case,
         * lifecycle management is disabled.
         */
        static PropertyManager createIfNotExists(MeshT &mesh, const char *propname) {
            PROPTYPE dummy_prop;
            PropertyManager pm(mesh, propname, mesh.get_property_handle(dummy_prop, propname));
            pm.retain();
            return std::move(pm);
        }


        PropertyManager duplicate(const char *clone_name) {
            PropertyManager pm(*mesh_, clone_name, false);
            pm.mesh_->property(pm.prop_) = mesh_->property(prop_);
            return std::move(pm);
        }

        /**
         * Included for backwards compatibility with non-C++11 version.
         */
        PropertyManager move() {
            return std::move(*this);
        }

#else
        class Proxy {
            private:
                Proxy(MeshT *mesh_, PROPTYPE prop_, bool retain_, const std::string &name_) :
                    mesh_(mesh_), prop_(prop_), retain_(retain_), name_(name_) {}
                MeshT *mesh_;
                PROPTYPE prop_;
                bool retain_;
                std::string name_;

                friend class PropertyManager;
        };

        operator Proxy() {
            Proxy p(mesh_, prop_, retain_, name_);
            mesh_ = 0;
            retain_ = true;
            return p;
        }

        Proxy move() {
            return (Proxy)*this;
        }

        PropertyManager(Proxy p) : mesh_(p.mesh_), prop_(p.prop_), retain_(p.retain_), name_(p.name_) {}

        PropertyManager &operator=(Proxy p) {
            PropertyManager(p).swap(*this);
            return *this;
        }

        /**
         * Create a property manager for the supplied property and mesh.
         * If the property doesn't exist, it is created. In any case,
         * lifecycle management is disabled.
         */
        static Proxy createIfNotExists(MeshT &mesh, const char *propname) {
            PROPTYPE dummy_prop;
            PropertyManager pm(mesh, propname, mesh.get_property_handle(dummy_prop, propname));
            pm.retain();
            return (Proxy)pm;
        }

        Proxy duplicate(const char *clone_name) {
            PropertyManager pm(*mesh_, clone_name, false);
            pm.mesh_->property(pm.prop_) = mesh_->property(prop_);
            return (Proxy)pm;
        }
#endif

        /**
         * \brief Disable lifecycle management for this property.
         *
         * If this method is called, the encapsulated property will not be deleted
         * upon destruction of the PropertyManager instance.
         */
        inline void retain(bool doRetain = true) {
            retain_ = doRetain;
        }

        /**
         * Access the encapsulated property.
         */
        inline PROPTYPE &operator* () {
            return prop_;
        }

        /**
         * Access the encapsulated property.
         */
        inline const PROPTYPE &operator* () const {
            return prop_;
        }

        /**
         * Enables convenient access to the encapsulated property.
         *
         * For a usage example see this class' documentation.
         *
         * @param handle A handle of the appropriate handle type. (I.e. \p VertexHandle for \p VPropHandleT, etc.)
         */
        template<typename HandleType>
        inline typename PROPTYPE::reference operator[] (const HandleType &handle) {
            return mesh_->property(prop_, handle);
        }

        /**
         * Enables convenient access to the encapsulated property.
         *
         * For a usage example see this class' documentation.
         *
         * @param handle A handle of the appropriate handle type. (I.e. \p VertexHandle for \p VPropHandleT, etc.)
         */
        template<typename HandleType>
        inline typename PROPTYPE::const_reference operator[] (const HandleType &handle) const {
            return mesh_->property(prop_, handle);
        }

        /**
         * Conveniently set the property for an entire range of values.
         *
         * Examples:
         * \code
         * MeshT mesh;
         * PropertyManager<VPropHandleT<double>, MeshT> distance(
         *     mesh, "distance.plugin-example.i8.informatik.rwth-aachen.de");
         * distance.set_range(
         *     mesh.vertices_begin(), mesh.vertices_end(),
         *     std::numeric_limits<double>::infinity());
         * \endcode
         * or
         * \code
         * MeshT::VertexHandle vh;
         * distance.set_range(
         *     mesh.vv_begin(vh), mesh.vv_end(vh),
         *     std::numeric_limits<double>::infinity());
         * \endcode
         *
         * @param begin Start iterator. Needs to dereference to HandleType.
         * @param end End iterator. (Exclusive.)
         * @param value The value the range will be set to.
         */
        template<typename HandleTypeIterator, typename PROP_VALUE>
        void set_range(HandleTypeIterator begin, HandleTypeIterator end,
                const PROP_VALUE &value) {
            for (; begin != end; ++begin)
                (*this)[*begin] = value;
        }

        /**
         * Conveniently transfer the values managed by one property manager
         * onto the values managed by a different property manager.
         *
         * @param begin Start iterator. Needs to dereference to HandleType. Will
         * be used with this property manager.
         * @param end End iterator. (Exclusive.) Will be used with this property
         * manager.
         * @param dst_propmanager The destination property manager.
         * @param dst_begin Start iterator. Needs to dereference to the
         * HandleType of dst_propmanager. Will be used with dst_propmanager.
         * @param dst_end End iterator. (Exclusive.)
         * Will be used with dst_propmanager. Used to double check the bounds.
         */
        template<typename HandleTypeIterator, typename PROPTYPE_2,
                 typename MeshT_2, typename HandleTypeIterator_2>
        void copy_to(HandleTypeIterator begin, HandleTypeIterator end,
                PropertyManager<PROPTYPE_2, MeshT_2> &dst_propmanager,
                HandleTypeIterator_2 dst_begin, HandleTypeIterator_2 dst_end) const {

            for (; begin != end && dst_begin != dst_end; ++begin, ++dst_begin) {
                dst_propmanager[*dst_begin] = (*this)[*begin];
            }
        }

        template<typename RangeType, typename PROPTYPE_2,
                 typename MeshT_2, typename RangeType_2>
        void copy_to(const RangeType &range,
                PropertyManager<PROPTYPE_2, MeshT_2> &dst_propmanager,
                const RangeType_2 &dst_range) const {
            copy_to(range.begin(), range.end(), dst_propmanager,
                    dst_range.begin(), dst_range.end());
        }

        /**
         * Copy the values of a property from a source range to
         * a target range. The source range must not be smaller than the
         * target range.
         *
         * @param prop_name Name of the property to copy. Must exist on the
         * source mesh. Will be created on the target mesh if it doesn't exist.
         *
         * @param src_mesh Source mesh from which to copy.
         * @param src_range Source range which to copy. Must not be smaller than
         * dst_range.
         * @param dst_mesh Destination mesh on which to copy.
         * @param dst_range Destination range.
         */
        template<typename RangeType, typename MeshT_2, typename RangeType_2>
        static void copy(const char *prop_name,
                MeshT &src_mesh, const RangeType &src_range,
                MeshT_2 &dst_mesh, const RangeType_2 &dst_range) {

            typedef OpenMesh::PropertyManager<PROPTYPE, MeshT> DstPM;
            DstPM dst(DstPM::createIfNotExists(dst_mesh, prop_name));

            typedef OpenMesh::PropertyManager<PROPTYPE, MeshT_2> SrcPM;
            SrcPM src(src_mesh, prop_name, true);

            src.copy_to(src_range, dst, dst_range);
        }

    private:
        void deleteProperty() {
            if (!retain_)
                mesh_->remove_property(prop_);
        }

    private:
        MeshT *mesh_;
        PROPTYPE prop_;
        bool retain_;
        std::string name_;
};

} /* namespace OpenMesh */
#endif /* PROPERTYMANAGER_HH_ */
