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
            void encodeRotation(T theta, T x, T y, T z)
            {
                w_ = cos(theta / 2);
                x_ = x * sin(theta / 2);
                y_ = y * sin(theta / 2);
                z_ = z * sin(theta / 2);
                normalize();
            }

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
            void getRotation( T& theta, T& x, T& y, T& z)
            {
                // Acquire the amount of rotation. Prevent against rounding error.
                if ((w_ > 1) || (w_ < -1))
                    theta = 2 * acos(1);
                else
                    theta = 2 * acos(w_);

                T commonVal = sin(theta /2);

                // Acquire rotational axis. Guard against division by 0.
                if (commonVal != 0)
                {
                    x = x_ / commonVal;
                    y = y_ / commonVal;
                    z = z_ / commonVal;
                }
                else // Guard against division by zero. Values are bogus but ignored.
                {
                    x = x_;
                    y = y_;
                    z = z_;
                }
            }


            /**
             * \fn void rotate( T& x, T& y, T& z)
             * \brief rotate a vector3 (x,y,z) by the angle theta about the axis
             * (U_x, U_y, U_z) stored in the quaternion.
             */
            void rotate(T& x, T& y, T& z)
            {
                Quaternion<T> q = (*this);
                Quaternion<T> qStar = (*this).conjugate();
                Quaternion<T> rotatedVal = q * Quaternion(0, x, y, z) * qStar;

                x = rotatedVal.x_;
                y = rotatedVal.y_;
                z = rotatedVal.z_;
            }

            /**
             * Quaternion Mathematical Properties
             * implemented below
             **/

            Quaternion operator+(const Quaternion& other)
            {
                return Quaternion(  (w_ + other.w_),
                                    (x_ + other.x_),
                                    (y_ + other.y_),
                                    (z_ + other.z_));
            }

            Quaternion operator-(const Quaternion& other)
            {
                return Quaternion((w_ - other.w_),
                                  (x_ - other.x_),
                                  (y_ - other.y_),
                                  (z_ - other.z_));
            }

            Quaternion operator-()
            {
                return Quaternion(-w_, -x_, -y_, -z_);
            }

            bool operator==(const Quaternion& other) const
            {
                return (w_ == other.w_) && (x_ == other.x_) && (y_ == other.y_) && (z_ == other.z_);
            }

            /**
             * \fn template <typename U> friend Quaternion operator*(const U scalar,
             *                                                       const Quaternion& q)
             * \brief implements scalar multiplication for arbitrary scalar types.
             */
            template <typename U> friend Quaternion operator*(const U scalar, const Quaternion& other)
            {
                return Quaternion<T>((scalar * other.w_),
                                     (scalar * other.x_),
                                     (scalar * other.y_),
                                     (scalar * other.z_));
            }

            template <typename U> friend Quaternion operator/(const Quaternion& q, const U scalar)
            {
                return (1 / scalar) * q;
            }

            /// Quaternion Product
            Quaternion operator*(const Quaternion& other)
            {
                return Quaternion(
                        ((w_*other.w_) - (x_*other.x_) - (y_*other.y_) - (z_*other.z_)),
                        ((w_*other.x_) + (x_*other.w_) + (y_*other.z_) - (z_*other.y_)),
                        ((w_*other.y_) - (x_*other.z_) + (y_*other.w_) + (z_*other.x_)),
                        ((w_*other.z_) + (x_*other.y_) - (y_*other.x_) + (z_*other.w_))
                );
            }

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
            Quaternion power(T exponent)
            {
                T magnitude = this->norm();

                Quaternion<T> unitQuaternion = *this;
                unitQuaternion.normalize();

                // unitQuaternion.w_ will always be less than 1, so no domain error.
                T theta = acos(unitQuaternion.w_);


                // Perform math:
                // N^exponent * [cos(exponent * theta)  + U*sin(exponent * theta)], where U is a vector.
                T poweredMag = pow(magnitude, exponent);  // N^exponent
                T cospTheta = cos(exponent * theta);
                T sinpTheta = sin(exponent * theta);

                return Quaternion( poweredMag * cospTheta,
                                   poweredMag * unitQuaternion.x_ * sinpTheta,
                                   poweredMag * unitQuaternion.y_ * sinpTheta,
                                   poweredMag * unitQuaternion.z_ * sinpTheta);
            }


            /**
             * \fn static T dotProduct(Quaternion q1, Quaternion q2)
             * \brief returns the dot product of two quaternions.
             */
            T dotProduct(Quaternion other)
            {
                return 0.5 * (conjugate() * other) + ((*this) * other.conjugate()).w_;
            }

            /// Conjugate
            Quaternion conjugate()
            {
                return Quaternion<T>(w_, -x_, -y_, -z_);
            }

            T norm()
            {
                return sqrt((w_ * w_) + (x_ * x_) + (y_ * y_) + (z_ * z_));
            }

            Quaternion inverse()
            {
                return (1/(*this).norm()) * (*this).conjugate();
            }

            /**
             * \fn void normalize()
             * \brief normalizes the quaternion to magnitude 1
             */
            void normalize()
            {
                // should never happen unless the Quaternion<T> wasn't initialized
                // correctly.
                assert( !((w_ == 0) && (x_ == 0) && (y_ == 0) && (z_ == 0)));
                T theNorm = norm();
                assert(theNorm > 0);
                (*this) = (1.0/theNorm) * (*this);
                return;
            }


            /**
             * \fn static Quaternion slerp( Quaternion q1 Quaternion q2,
             *                                 T percentage)
             * \brief return a quaternion that is a linear interpolation between q1 and q2
             *        where percentage (from 0 to 1) defines the amount of interpolation
             * \details morph one quaternion into the other with constant 'velocity.'
             *          Implementation details from Wikipedia article on Slerp.
             */
            Quaternion slerp(Quaternion other, double t)
            {
                // Only unit quaternions are valid rotations.
                // Normalize to avoid undefined behavior.
                normalize();
                other.normalize();

                // Compute the cosine of the angle between the two vectors.
                double dot = dotProduct(other);

                const double DOT_THRESHOLD = 0.9995;
                if (fabs(dot) > DOT_THRESHOLD) {
                    // If the inputs are too close for comfort, linearly interpolate
                    // and normalize the result.

                    Quaternion<T> result = *this + t*(other - *this);
                    result.normalize();
                    return result;
                }

                // If the dot product is negative, the quaternions
                // have opposite handed-ness and slerp won't take
                // the shorter path. Fix by reversing one quaternion.
                if (dot < 0.0f) {
                    other = -other;
                    dot = -dot;
                }

                //            Clamp(dot, -1, 1);           // Robustness: Stay within domain of acos()
                double theta_0 = acos(dot);  // theta_0 = angle between input vectors
                double theta = theta_0*t;    // theta = angle between v0 and result

                Quaternion<T> v2 = other - dot * (*this);
                v2.normalize();              // { v0, v2 } is now an orthonormal basis

                return cos(theta)* (*this) + sin(theta) * v2;
            }

            /**
             * \fn template <typename U> friend std::ostream& operator <<
             *                                  (std::ostream& os, const Quaternion<U>& q);
             * \brief a templated friend function for printing quaternions.
             * \details T cannot be used as dummy parameter since it would be shared by
             *          the class, and this function is not a member function.
             */
            template <typename U> friend std::ostream& operator << (std::ostream& os, const Quaternion<U>& q)
            {
                os << "(" << q.w_ << ", " << q.x_ << ", " <<  q.y_ << ", " << q.z_ << ")";
                return os;
            }

            T w_;
            T x_;
            T y_;
            T z_;
        };
    }
}
#endif // DYNAMIC_FUSION_QUATERNION_HPP