/**
 * Adapted from https://github.com/Poofjunior/QPose
 */
#include <iostream>
#include "quaternion.hpp"
#include <ctgmath>
#include <math.h>
namespace kfusion {
    namespace utils {


        Quaternion::Quaternion() : w_(1), x_(0), y_(0), z_(0) {}

        template <typename T>
        Quaternion::Quaternion( T w, T x, T y, T z) : w_(w), x_(x), y_(y), z_(z) {}

        Quaternion::~Quaternion() {}

        template <typename T>
        void Quaternion::encodeRotation( T theta, T x, T y, T z)
        {
            w_ = cos(theta / 2);
            x_ = x * sin(theta / 2);
            y_ = y * sin(theta / 2);
            z_ = z * sin(theta / 2);
            normalize();
        }

        template <typename T>
        void Quaternion::getRotation( T& theta, T& x, T& y, T& z)
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

        template <typename T>
        void Quaternion::rotate(T& x, T& y, T& z)
        {
            Quaternion q = (*this);
            Quaternion qStar = (*this).conjugate();
            Quaternion rotatedVal = q * Quaternion(0, x, y, z) * qStar;

            x = rotatedVal.x_;
            y = rotatedVal.y_;
            z = rotatedVal.z_;
        }


        Quaternion Quaternion::operator+(const Quaternion& other)
        {
            return Quaternion(  (w_ + other.w_),
                                (x_ + other.x_),
                                (y_ + other.y_),
                                (z_ + other.z_));
        }

        Quaternion Quaternion::operator-(const Quaternion& q2)
        {
            return Quaternion(  (w_ - q2.w_),
                                (x_ - q2.x_),
                                (y_ - q2.y_),
                                (z_ - q2.z_));
        }
        Quaternion Quaternion::operator-()
        {
            return Quaternion(-w_, -x_, -y_, -z_);
        }

        bool Quaternion::operator==(const Quaternion& other) const
        {
            return (w_ == other.w_) && (x_ == other.x_)
                   && (y_ == other.y_) && (z_ == other.z_);
        }

        template <typename U> friend Quaternion Quaternion::operator*(const U scalar, const Quaternion& other)
        {
            return Quaternion(  (scalar * other.w_),
                                (scalar * other.x_),
                                (scalar * other.y_),
                                (scalar * other.z_));
        }

        template <typename U> friend Quaternion Quaternion::operator/(const Quaternion& q, const U scalar)
        {
            return (1 / scalar) * other;
        }

        Quaternion Quaternion::operator*(const Quaternion& other)
        {
            return Quaternion(
                    ((w_*other.w_) - (x_*other.x_) - (y_*other.y_) - (z_*other.z_)),
                    ((w_*other.x_) + (x_*other.w_) + (y_*other.z_) - (z_*other.y_)),
                    ((w_*other.y_) - (x_*other.z_) + (y_*other.w_) + (z_*other.x_)),
                    ((w_*other.z_) + (x_*other.y_) - (y_*other.x_) + (z_*other.w_))
            );
        }

        template <typename T>
        Quaternion Quaternion::power(T p)
        {
            T magnitude = this->norm();

            Quaternion unitQuaternion = *this;
            unitQuaternion.normalize();

            // unitQuaternion.w_ will always be less than 1, so no domain
            // error.
            T theta = acos(unitQuaternion.w_);


            // Perform math:
            // N^p * [cos(p * theta)  + U*sin(p * theta)], where U is a vector.
            T poweredMag = pow(magnitude, p);  // N^p
            T cospTheta = cos(p * theta);
            T sinpTheta = sin(p * theta);
/*
            std::cout << "poweredMag is " << poweredMag << std::endl;
            std::cout << "cospTheta is " << cospTheta << std::endl;
            std::cout << "p * Theta is " << p * theta << std::endl;
            std::cout << "sinpTheta is " << sinpTheta << std::endl;
*/

            // Note: U_x, U_y, U_z exist in normalized q1.

            return Quaternion( poweredMag * cospTheta,
                               poweredMag * unitQuaternion.x_ * sinpTheta,
                               poweredMag * unitQuaternion.y_ * sinpTheta,
                               poweredMag * unitQuaternion.z_ * sinpTheta);
        }

        template <typename T>
        T Quaternion::dotProduct(Quaternion other)
        {
            return 0.5 * (conjugate() * other) + ((*this) * other.conjugate()).w_;
        }

        Quaternion Quaternion::conjugate()
        {
            return Quaternion(  w_, -x_, -y_, -z_);
        }

        template <typename T>
        T Quaternion::norm()
        {
            return sqrt((w_ * w_) + (x_ * x_) + (y_ * y_) + (z_ * z_));
        }

        Quaternion Quaternion::inverse()
        {
            return (1/(*this).norm()) * (*this).conjugate();
        }

        template <typename T>
        void Quaternion::normalize()
        {
            // should never happen unless the quaternion wasn't initialized
            // correctly.
//            assert( !((w_ == 0) && (x_ == 0) && (y_ == 0) && (z_ == 0)));
            T theNorm = norm();
//            assert(theNorm > 0);
            (*this) = (1.0/theNorm) * (*this);
            return;
        }


        Quaternion Quaternion::slerp(Quaternion other, double t) {
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

                Quaternion result = *this + t*(other - *this);
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

            Quaternion v2 = other - dot * (*this);
            v2.normalize();              // { v0, v2 } is now an orthonormal basis

            return cos(theta)* (*this) + sin(theta) * v2;
        }
        template <typename U> friend std::ostream& operator<<(std::ostream& os, const Quaternion<U>& q)
        {
            os << "(" << q.w_ << ", " << q.x_ << ", " <<  q.y_ << ", " << q.z_ << ")";
            return os;
        }
    };

}
