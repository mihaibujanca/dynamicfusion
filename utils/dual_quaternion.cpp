/**
 * Adapted from https://github.com/Poofjunior/QPose
 */
#include "dual_quaternion.hpp"
#include "quaternion.hpp"
#include <math.h>
namespace kfusion
{
    namespace utils
    {
        template<typename T>
        DualQuaternion::DualQuaternion(){};

        template<typename T>
        DualQuaternion::DualQuaternion(T x, T y, T z, T roll, T pitch, T yaw   )
        {
            // convert here.
            rotation_.w_ = cos(roll / 2) * cos(pitch / 2) * cos(yaw / 2) +
                           sin(roll / 2) * sin(pitch / 2) * sin(yaw / 2);
            rotation_.x_ = sin(roll / 2) * cos(pitch / 2) * cos(yaw / 2) -
                           cos(roll / 2) * sin(pitch / 2) * sin(yaw / 2);
            rotation_.y_ = cos(roll / 2) * sin(pitch / 2) * cos(yaw / 2) +
                           sin(roll / 2) * cos(pitch / 2) * sin(yaw / 2);
            rotation_.z_ = cos(roll / 2) * cos(pitch / 2) * sin(yaw / 2) -
                           sin(roll / 2) * cos(pitch / 2) * cos(yaw / 2);

            translation_ = 0.5 * Quaternion<T>(0, x, y, z) * rotation_;
        }

        template<typename T>
        DualQuaternion::DualQuaternion(Quaternion<T> translation, Quaternion<T> rotation)
        {
            rotation_ = rotation;
            translation_ = 0.5 * translation * rotation;
        }

        DualQuaternion::~DualQuaternion(){}

        template<typename T>
        void DualQuaternion::encodeRotation(T angle, T x, T y, T z)
        {
            rotation_.encodeRotation(angle, x, y, z);
        }

        template<typename T>
        void DualQuaternion::encodeTranslation(T x, T y, T z)
        {
            translation_ = 0.5 * Quaternion<T>(0, x, y, z) * rotation_;
        }

        template<typename T>
        void DualQuaternion::normalizeRotation()
        {
            T x, y, z;
            getTranslation(x, y, z);

            rotation_.normalize();

            encodeTranslation(x, y, z);
        }

        template<typename T>
        void DualQuaternion::getTranslation(T& x, T& y, T& z)
        {
            Quaternion<T> result = 2 * translation_ * rotation_.conjugate();
            /// note: inverse of a quaternion is the same as the conjugate.
            x = result.x_;
            y = result.y_;
            z = result.z_;
        }

        template<typename T>
        Quaternion<T> DualQuaternion::getTranslation()
        {
            return 2 * translation_ * rotation_.conjugate();
        }

        template<typename T>
        void DualQuaternion::getEuler( T& roll, T& pitch, T& yaw)
        {
/// FIXME: breaks for some value around PI.
            roll = getRoll();
            pitch = getPitch();
            yaw = getYaw();
        }

        template<typename T>
        Quaternion<T> DualQuaternion::getRotation()
        {
            return rotation_;
        }

        template <typename T>
        void DualQuaternion::get6DOF(T& x, T& y, T& z, T& roll, T& pitch, T& yaw)
        {
            getTranslation(x, y, z);
            getEuler(roll, pitch, yaw);
        }

        DualQuaternion DualQuaternion::operator+(const DualQuaternion& other)
        {
            DualQuaternion result;
            result.rotation_ = rotation_ + other.rotation_;
            result.translation_ = translation_ + other.translation_;
            return result;
        }

        DualQuaternion DualQuaternion::operator-(const DualQuaternion& other)
        {
            DualQuaternion result;
            result.rotation_ = rotation_ - other.rotation_;
            result.translation_ = translation_ - other.translation_;
            return result;
        }

        template <typename U>
        friend DualQuaternion DualQuaternion::operator*(const U scalar, const DualQuaternion& q)
        {
            DualQuaternion result;
            // Luckily, left-scalar multiplication is implemented for
            // Quaternions.
            result.rotation_ = scalar * q.rotation_;
            result.translation_ = scalar * q.translation_;
            return result;
        }

        DualQuaternion DualQuaternion::operator*(const DualQuaternion& other)
        {
            DualQuaternion result;
            result.rotation_ = rotation_ * other.rotation_;
            result.translation_ = (rotation_ * other.translation_) + (translation_ * other.rotation_);
            return result;
        }

        template <typename T>
        DualQuaternion DualQuaternion::operator/(const std::tuple<T,T> divisor)
        {
            DualQuaternion result;
            result.rotation_ = 1 / divisor * rotation_;
            result.translation_ = 1/ divisor * translation_;
            return result;
        }

        DualQuaternion DualQuaternion::conjugate()
        {
            DualQuaternion result;
            result.rotation_ = rotation_.conjugate();
            result.translation_ = translation_.conjugate();
            return result;
        }


        template <typename T>
        std::pair<T, T> DualQuaternion::magnitude()
        {
            DualQuaternion result = (*this) * (*this).conjugate();
//            TODO: only print when debugging
            std::cout << result.rotation_;
            std::cout << result.translation_;
            return std::make_pair(result.rotation_.w_, result.translation_.w_);
        }

        template <typename T>
        T DualQuaternion::getRoll()
        {
            // TODO: verify this!
            return atan2(2*((rotation_.w_ * rotation_.x_) + (rotation_.y_ * rotation_.z_)),
                         (1 - 2*((rotation_.x_*rotation_.x_) + (rotation_.y_*rotation_.y_))));
        }

        template <typename T>
        T DualQuaternion::getPitch()
        {
            // TODO: verify this!
            return asin(2*(rotation_.w_ * rotation_.y_ - rotation_.z_ * rotation_.x_));
        }

        template <typename T>
        T DualQuaternion::getYaw()
        {
            // TODO: verify this!
            return atan2(2*((rotation_.w_ * rotation_.z_) + (rotation_.x_ * rotation_.y_)),
                         (1 - 2*((rotation_.y_*rotation_.y_) + (rotation_.z_*rotation_.z_))));
        }

    template <typename T> std::ostream& DualQuaternion::operator<< (std::ostream& os, const DualQuaternion<T>& q)
    {
        os << "[" << q.rotation_ << ", " << q.translation_ << ", " << "]" << std::endl;
        return os;
    }

}
}