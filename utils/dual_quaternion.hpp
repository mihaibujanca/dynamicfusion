#ifndef DYNAMIC_FUSION_DUAL_QUATERNION_HPP
#define DYNAMIC_FUSION_DUAL_QUATERNION_HPP
#include<iostream>
#include<quaternion.hpp>
//TODO: Quaternion class can be forward declared rather than included
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
            DualQuaternion();

            /**
             * \brief constructor that takes cartesian coordinates and Euler angles as
             *        arguments.
             */
            DualQuaternion(T x, T y, T z, T roll, T pitch, T yaw);

            /**
             * \brief constructor that takes two quaternions as arguments.
             * \details The rotation
             *          quaternion has the conventional encoding for a rotation as a
             *          quaternion. The translation quaternion is a quaternion with
             *          cartesian coordinates encoded as (0, x, y, z)
             */
            DualQuaternion(Quaternion<T> translation, Quaternion<T> rotation);
            // Other constructors here.

            ~DualQuaternion();

            /**
             * \brief store a rotation
             * \param angle is in radians
             */
            void encodeRotation(T angle, T x, T y, T z);

            void encodeTranslation(T x, T y, T z);

            /// handle accumulating error.
            void normalizeRotation();


            /**
             * \brief a reference-based method for acquiring the latest
             *        translation data.
             */
            void getTranslation(T &x, T &y, T &z);

            Quaternion<T> getTranslation();

            /**
             * \brief a reference-based method for acquiring the latest rotation data.
             */
            void getEuler(T &roll, T &pitch, T &yaw);

            Quaternion<T> getRotation();


            /**
             * \brief Extraction everything (in a nice format)
             */
            void get6DOF(T &x, T &y, T &z, T &roll, T &pitch, T &yaw);
            DualQuaternion operator+(const DualQuaternion &other);
            DualQuaternion operator-(const DualQuaternion &other);
            DualQuaternion operator*(const DualQuaternion &other);
            DualQuaternion operator/(const std::pair<T,T> divisor);
            /// (left) Scalar Multiplication
            /**
             * \fn template <typename U> friend Quaternion operator*(const U scalar,
             * \brief implements scalar multiplication for arbitrary scalar types
             */
            template<typename U>
            friend DualQuaternion operator*(const U scalar, const DualQuaternion &q);
            DualQuaternion conjugate();
            std::pair<T,T> magnitude(); // TODO: check if this works

        private:
            Quaternion<T> rotation_;
            Quaternion<T> translation_;

            T position_[3] = {};    /// default initialize vector to zeros.

            T rotAxis_[3] = {};     /// default initialize vector to zeros.
            T rotAngle_;


            T getRoll();
            T getPitch();
            T getYaw();
        };
        template <typename T>
        std::ostream &operator<<(std::ostream &os, const DualQuaternion<T> &q)
        {
            os << "[" << q.rotation_ << ", " << q.translation_ << ", " << "]" << std::endl;
            return os;
        }
    }
}
#endif //DualQuaternion_HPP