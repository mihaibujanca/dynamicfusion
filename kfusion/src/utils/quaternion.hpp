/**
 */
#ifndef DYNAMIC_FUSION_QUATERNION_HPP
#define DYNAMIC_FUSION_QUATERNION_HPP

#include <iostream>

namespace kfusion{
    namespace utils{


        /**
         * \class Quaternion
         * \brief a templated quaternion class that also enables quick storage and
         *        retrieval of rotations encoded as a vector3 and angle.
         * \details All angles are in radians.
         * \warning This template is intended to be instantiated with a floating point
         *          data type.
         */
        template <typename T> class Quaternion
        {
        public:
            Quaternion() : w_(1), x_(0), y_(0), z_(0)
            {}

            Quaternion(T w, T x, T y, T z) : w_(w), x_(x), y_(y), z_(z)
            {}

            ~Quaternion()
            {}


            /**
             * Quaternion Rotation Properties for straightforward usage of quaternions
             *  to store rotations.
             */

            /**
             * \fn void encodeRotation( T theta, T x, T y, T z)
             * \brief Store a normalized rotation in the quaternion encoded as a rotation
             *        of theta about the vector (x,y,z).
             */
            void encodeRotation(T theta, T x, T y, T z);

            /**
             * \fn void getRotation( T& angle, T& x, T& y, T& z)
             * \brief Retrieve the rotation (angle and vector3) stored in the quaternion.
             * \warning only unit quaternions represent rotation.
             * \details A quaternion:
             *          Q = cos(alpha) + Usin(alpha), where U is a vector3, stores a
                        rotation
             *          of 2*alpha about the 3D axis U. This member function retrieves
                        theta and U, where theta = 2*alpha is the amount of rotation
                        about the vector U.
             * \note the angle retrieved is in radians.
             */
            void getRotation( T& theta, T& x, T& y, T& z);


            /**
             * \fn void rotate( T& x, T& y, T& z)
             * \brief rotate a vector3 (x,y,z) by the angle theta about the axis
             * (U_x, U_y, U_z) stored in the quaternion.
             */
            void rotate(T& x, T& y, T& z);
            /**
             * Quaternion Mathematical Properties
             * implemented below */

            Quaternion operator+(const Quaternion& q2);
            Quaternion operator-(const Quaternion& q2);
            Quaternion operator-();
            bool operator==(const Quaternion& other) const;
            /**
             * \fn template <typename U> friend Quaternion operator*(const U scalar,
             *                                                       const Quaternion& q)
             * \brief implements scalar multiplication for arbitrary scalar types.
             */
            template <typename U> friend Quaternion operator*(const U scalar, const Quaternion& q);
            template <typename U> friend Quaternion operator/(const Quaternion& q, const U scalar);

            /// Quaternion Product
            Quaternion operator*(const Quaternion& other);


            /// Quaternion Power function
            /**
             * \fn static Quaternion power(const Quaternion q1, T p)
             * \brief perform the power operation on a quaternion
             * \details A quaternion Q = (w, x, y, z) may be written as the
             * product of a scalar and a unit quaternion: Q = N*q =
             * N[sin(theta) + U_x*cos(theta) + U_y*cos(theta) + U_k*cos(theta)], where N is
             * a scalar and U is a vector3 (U_x, U_y, U_z) representing the normalized
             * vector component of the original quaternion, aka: (x,y,z). Raising a
             * quaternion to a p._v.w*_v.y - rhs._v.x*_v.z + rhs._v.y*_v.w + rhs._v.z*_v.x,wer can be done most easily in this form.
             */
            Quaternion power(T p);

            /**
             * \fn static T dotProduct(Quaternion q1, Quaternion q2)
             * \brief returns the dot product of two quaternions.
             */
            T dotProduct(Quaternion other);

            /// Conjugate
            Quaternion conjugate()
            {
                return Quaternion(  w_, -x_, -y_, -z_);
            }

            T norm();
            Quaternion inverse();

            /**
             * \fn void normalize()
             * \brief normalizes the quaternion to magnitude 1
             */
            void normalize();


            /**
             * \fn static Quaternion slerp( Quaternion q1 Quaternion q2,
             *                                 T percentage)
             * \brief return a quaternion that is a linear interpolation between q1 and q2
             *        where percentage (from 0 to 1) defines the amount of interpolation
             * \details morph one quaternion into the other with constant 'velocity.'
             *          Implementation details from Wikipedia article on Slerp.
             */
            Quaternion slerp(Quaternion other, double t);

            /**
             * \fn template <typename U> friend std::ostream& operator <<
             *                                  (std::ostream& os, const Quaternion<U>& q);
             * \brief a templated friend function for printing quaternions.
             * \details T cannot be used as dummy parameter since it would be shared by
             *          the class, and this function is not a member function.
             */
            template <typename U> friend std::ostream& operator << (std::ostream& os, const Quaternion<U>& q);

            T w_;
            T x_;
            T y_;
            T z_;
        };
    }
}
#endif // DYNAMIC_FUSION_QUATERNION_HPP