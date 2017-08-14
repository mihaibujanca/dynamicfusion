#ifndef DYNAMIC_FUSION_DUAL_QUATERNION_HPP
#define DYNAMIC_FUSION_DUAL_QUATERNION_HPP
#include<iostream>
#include<quaternion.hpp>
//TODO: Quaternion class can be forward declared rather than included
//Adapted from https://github.com/Poofjunior/QPose
/**
 * \brief a dual quaternion class for encoding transformations.
 * \details transformations are stored as first a translation; then a
 *          rotation. It is possible to switch the order. See this paper:
 *  https://www.thinkmind.org/download.php?articleid=intsys_v6_n12_2013_5
 */
namespace kfusion {
    namespace utils {
        template<typename T>
        class DualQuaternion {
        public:
            /**
             * \brief default constructor.
             */
            DualQuaternion(){};

            /**
             * \brief constructor that takes cartesian coordinates and Euler angles as
             *        arguments.
             */
            DualQuaternion(T x, T y, T z, T roll, T pitch, T yaw)
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

            /**
             * \brief constructor that takes two quaternions as arguments.
             * \details The rotation
             *          quaternion has the conventional encoding for a rotation as a
             *          quaternion. The translation quaternion is a quaternion with
             *          cartesian coordinates encoded as (0, x, y, z)
             */
            DualQuaternion(Quaternion<T> translation, Quaternion<T> rotation)
            {
                rotation_ = rotation;
                translation_ = 0.5 * translation * rotation;
            }

            ~DualQuaternion(){};

            /**
             * \brief store a rotation
             * \param angle is in radians
             */
            void encodeRotation(T angle, T x, T y, T z)
            {
                rotation_.encodeRotation(angle, x, y, z);
            }

            void encodeTranslation(T x, T y, T z)
            {
                translation_ = 0.5 * Quaternion<T>(0, x, y, z) * rotation_;
            }

            /// handle accumulating error.
            void normalizeRotation()
            {
                T x, y, z;
                getTranslation(x, y, z);

                rotation_.normalize();

                encodeTranslation(x, y, z);
            }

            /**
             * \brief a reference-based method for acquiring the latest
             *        translation data.
             */
            void getTranslation(T &x, T &y, T &z) const
            {
                Quaternion<T> result = 2 * translation_ * rotation_.conjugate();
                /// note: inverse of a quaternion is the same as the conjugate.
                x = result.x_;
                y = result.y_;
                z = result.z_;
            }

            /**
             * \brief a reference-based method for acquiring the latest
             *        translation data.
             */
            void getTranslation(Vec3f& vec3f) const
            {
                Quaternion<T> result = 2 * translation_ * rotation_.conjugate();
                vec3f = Vec3f(result.x_, result.y_, result.z_);
            }

            Quaternion<T> getTranslation() const
            {
                return 2 * translation_ * rotation_.conjugate();
            }


            /**
             * \brief a reference-based method for acquiring the latest rotation data.
             */
            void getEuler(T &roll, T &pitch, T &yaw)
            {
                // FIXME: breaks for some value around PI.
                roll = getRoll();
                pitch = getPitch();
                yaw = getYaw();
            }

            Quaternion<T> getRotation() const
            {
                return rotation_;
            }


            /**
             * \brief Extraction everything (in a nice format)
             */
            void get6DOF(T &x, T &y, T &z, T &roll, T &pitch, T &yaw)
            {
                getTranslation(x, y, z);
                getEuler(roll, pitch, yaw);
            }

            DualQuaternion operator+(const DualQuaternion &other)
            {
                DualQuaternion result;
                result.rotation_ = rotation_ + other.rotation_;
                result.translation_ = translation_ + other.translation_;
                return result;
            }

            DualQuaternion operator-(const DualQuaternion &other)
            {
                DualQuaternion result;
                result.rotation_ = rotation_ - other.rotation_;
                result.translation_ = translation_ - other.translation_;
                return result;
            }

            DualQuaternion operator*(const DualQuaternion &other)
            {
                DualQuaternion<T> result;
                result.rotation_ = rotation_ * other.rotation_;
                result.translation_ = (rotation_ * other.translation_) + (translation_ * other.rotation_);
                return result;
            }

            DualQuaternion operator/(const std::pair<T,T> divisor)
            {
                DualQuaternion<T> result;
                result.rotation_ = 1 / divisor.first * rotation_;
                result.translation_ = 1 / divisor.second * translation_;
                return result;
            }

            /// (left) Scalar Multiplication
            /**
             * \fn template <typename U> friend Quaternion operator*(const U scalar,
             * \brief implements scalar multiplication for arbitrary scalar types
             */
            template<typename U>
            friend DualQuaternion operator*(const U scalar, const DualQuaternion &q)
            {
                DualQuaternion<T> result;
                result.rotation_ = scalar * q.rotation_;
                result.translation_ = scalar * q.translation_;
                return result;
            }

            DualQuaternion conjugate()
            {
                DualQuaternion<T> result;
                result.rotation_ = rotation_.conjugate();
                result.translation_ = translation_.conjugate();
                return result;
            }

            void transform(Vec3f& point) // TODO: this should be a lot more generic
            {

            }

            std::pair<T,T> magnitude()
            {
                DualQuaternion result = (*this) * (*this).conjugate();
                // TODO: only print when debugging
//                std::cout << result.rotation_;
//                std::cout << result.translation_;
                return std::make_pair(result.rotation_.w_, result.translation_.w_);
            }

        private:
            Quaternion<T> rotation_;
            Quaternion<T> translation_;

            T position_[3] = {};    /// default initialize vector to zeros.

            T rotAxis_[3] = {};     /// default initialize vector to zeros.
            T rotAngle_;


            T getRoll()
            {
                // TODO: test this!
                return atan2(2*((rotation_.w_ * rotation_.x_) + (rotation_.y_ * rotation_.z_)),
                             (1 - 2*((rotation_.x_*rotation_.x_) + (rotation_.y_*rotation_.y_))));
            }

            T getPitch()
            {
                // TODO: test this!
                return asin(2*(rotation_.w_ * rotation_.y_ - rotation_.z_ * rotation_.x_));
            }

            T getYaw()
            {
                // TODO: test this!
                return atan2(2*((rotation_.w_ * rotation_.z_) + (rotation_.x_ * rotation_.y_)),
                             (1 - 2*((rotation_.y_*rotation_.y_) + (rotation_.z_*rotation_.z_))));
            }
        };

        template <typename T>
        std::ostream &operator<<(std::ostream &os, const DualQuaternion<T> &q)
        {
            os << "[" << q.getRotation() << ", " << q.getTranslation()<< ", " << "]" << std::endl;
            return os;
        }
    }
}
#endif //DYNAMIC_FUSION_DUAL_QUATERNION_HPP