#ifndef DYNAMIC_FUSION_QUATERNION_CU_HPP
#define DYNAMIC_FUSION_QUATERNION_CU_HPP

#include <host_defines.h>
#include "dualquaternion/transfo.hpp"

namespace kfusion {

class __align__(16) Quaternion_cuda{
    public:

    // -------------------------------------------------------------------------
    /// @name Constructors
    // -------------------------------------------------------------------------

    /// Default constructor : build a zero rotation.
	__device__ __host__ Quaternion_cuda()
    {
        w_ = 1.f;
        x_ = 0.f; y_ = 0.f; z_ = 0.f;
    }

    /// Copy constructor
	__device__ __host__ Quaternion_cuda(const Quaternion_cuda& q){
        w_ = q.w();
        x_ = q.i(); y_ = q.j(); z_ = q.k();
    }

    /// directly fill the quaternion
	__device__ __host__ Quaternion_cuda(float w, float i, float j, float k){
        w_ = w;
        x_ = i; y_ = j; z_ = k;
    }

    /// directly fill the quaternion vector and scalar part
	__device__ __host__ Quaternion_cuda(float w, const Vec3& v){
        w_ = w;
        x_ = v.x; y_ = v.y; z_ = v.z;
    }

    /// Construct the quaternion from the transformation matrix 't'
    /// Translation of 't' is ignored as quaternions can't represent it
	/// NOTE: normalize is performed inside
	__device__ __host__ Quaternion_cuda(const Transfo& t)
    {
        // Compute trace of matrix 't'
        float T = 1 + t.m[0] + t.m[5] + t.m[10];

        float S, X, Y, Z, W;

        if ( T > 0.00000001f ) // to avoid large distortions!
        {
            S = sqrt(T) * 2.f;
            X = ( t.m[6] - t.m[9] ) / S;
            Y = ( t.m[8] - t.m[2] ) / S;
            Z = ( t.m[1] - t.m[4] ) / S;
            W = 0.25f * S;
        }
        else
        {
            if ( t.m[0] > t.m[5] && t.m[0] > t.m[10] )
            {
                // Column 0 :
                S  = sqrt( 1.0f + t.m[0] - t.m[5] - t.m[10] ) * 2.f;
                X = 0.25f * S;
                Y = (t.m[1] + t.m[4] ) / S;
                Z = (t.m[8] + t.m[2] ) / S;
                W = (t.m[6] - t.m[9] ) / S;
            }
            else if ( t.m[5] > t.m[10] )
            {
                // Column 1 :
                S  = sqrt( 1.0f + t.m[5] - t.m[0] - t.m[10] ) * 2.f;
                X = (t.m[1] + t.m[4] ) / S;
                Y = 0.25f * S;
                Z = (t.m[6] + t.m[9] ) / S;
                W = (t.m[8] - t.m[2] ) / S;
            }
            else
            {   // Column 2 :
                S  = sqrt( 1.0f + t.m[10] - t.m[0] - t.m[5] ) * 2.f;
                X = (t.m[8] + t.m[2] ) / S;
                Y = (t.m[6] + t.m[9] ) / S;
                Z = 0.25f * S;
                W = (t.m[1] - t.m[4] ) / S;
            }
        }

        w_ = W; x_ = -X; y_ = -Y; z_ = -Z;
		normalize();
    }


    /// Construct the quaternion from the a rotation axis 'axis' and the angle
    /// 'angle' in radians
	__device__ __host__ Quaternion_cuda(const Vec3& axis, float angle)
    {
        Vec3 vec_axis = axis.normalized();
        float sin_a = sin( angle * 0.5f );
        float cos_a = cos( angle * 0.5f );
        w_    = cos_a;
        x_    = vec_axis.x * sin_a;
        y_    = vec_axis.y * sin_a;
        z_    = vec_axis.z * sin_a;
        // It is necessary to normalize the quaternion in case any values are
        // very close to zero.
        normalize();
    }

    // -------------------------------------------------------------------------
    /// @name Methods
    // -------------------------------------------------------------------------

    /// The conjugate of a quaternion is the inverse rotation
    /// (when the quaternion is normalized
	__device__ __host__ Quaternion_cuda conjugate() const
    {
        return Quaternion_cuda( w_, -x_,
                       -y_, -z_);
    }

    // TODO: Construct the quaternion from the rotation axis 'vec' and the
    // angle 'angle'
    // Quaternion_cuda(const Vec3& vec, float angle)

    /// Do the rotation of vector 'v' with the quaternion
	__device__ __host__ Vec3 rotate(const Vec3& v) const
    {

        // The conventionnal way to rotate a vector
        /*
        Quaternion_cuda tmp = *this;
        tmp.normalize();
        // Compute the quaternion inverse with
        Quaternion_cuda inv = tmp.conjugate();
        // Compute q * v * inv; in order to rotate the vector v
        // to do so v must be expressed as the quaternion q(0, v.x, v.y, v.z)
        return (Vec3)(*this * Quaternion_cuda(0, v) * inv);
        */

        // An optimized way to compute rotation
        Vec3 q_vec = get_vec_part();
        return v + (q_vec*2.f).cross( q_vec.cross(v) + v*w_ );
    }

    /// Do the rotation of point 'p' with the quaternion
	__device__ __host__ Point3 rotate(const Point3& p) const
    {
        Vec3 v = rotate((Vec3)p);
        return Point3(v.x, v.y, v.z);
    }

    /// Convert the quaternion to a rotation matrix
    /// @warning don't forget to normalize it before conversion
	__device__ __host__ Mat3 to_matrix3()const
    {
        float W = w_, X = -x_, Y = -y_, Z = -z_;
        float xx = X * X, xy = X * Y, xz = X * Z, xw = X * W;
        float yy = Y * Y, yz = Y * Z, yw = Y * W, zz = Z * Z;
        float zw = Z * W;
        Mat3 mat = Mat3(
                    1.f - 2.f * (yy + zz),      2.f * (xy + zw),       2.f * (xz - yw),
                          2.f * (xy - zw),1.f - 2.f * (xx + zz),       2.f * (yz + xw),
                          2.f * (xz + yw),      2.f * (yz - xw), 1.f - 2.f * (xx + yy)
                    );

        return mat;
    }

	__device__ __host__ void to_angleAxis(Vec3& axis, float& angle)const
	{
		const float scale = get_vec_part().norm();

		if (scale == float(0) || w() > float(1) || w() < float(-1))
		{
			angle = 0.0f;
			axis[0] = 0.0f;
			axis[1] = 1.0f;
			axis[2] = 0.0f;
		}
		else
		{
			angle = float(2.0) * acosf(w());
			axis = get_vec_part() / scale;
		}
	}

	__device__ __host__ Vec3 get_vec_part() const
    {
        return Vec3(x_, y_, z_);
    }

	__device__ __host__ float norm() const
    {
        return sqrt(w_*w_ +
                    x_*x_ +
                    y_*y_ +
                    z_*z_);
    }

	__device__ __host__ float normalize()
    {
        float n = norm();
        w_ /= n;
        x_ /= n;
        y_ /= n;
        z_ /= n;
        return n;
    }

	__device__ __host__ float& operator[](int i)
	{
#if 0
		switch (i)
		{
		case 0:
			return w_;
		case 1:
			return x_;
		case 2:
			return y_;
		case 3:
			return z_;
		default:
			return z_;
		}
#else
		return ((float*)this)[i];
#endif
	}

	__device__ __host__ const float& operator[](int i)const
	{
#if 0
		switch (i)
		{
		case 0:
			return w_;
		case 1:
			return x_;
		case 2:
			return y_;
		case 3:
			return z_;
		default:
			return z_;
		}
#else
		return ((float*)this)[i];
#endif
	}

	__device__ __host__ float dot(const Quaternion_cuda& q){
        return w() * q.w() + i() * q.i() + j() * q.j() + k() * q.k();
    }

	__device__ __host__ float w() const { return w_; }
	__device__ __host__ float i() const { return x_; }
	__device__ __host__ float j() const { return y_; }
	__device__ __host__ float k() const { return z_; }

    // -------------------------------------------------------------------------
    /// @name Operators
    // -------------------------------------------------------------------------

	__device__ __host__ Quaternion_cuda operator/ (float scalar) const
    {
        Quaternion_cuda q = *this;
        q.w_ /= scalar;
        q.x_ /= scalar;
        q.y_ /= scalar;
        q.z_ /= scalar;
        return q;
    }

	__device__ __host__ Quaternion_cuda operator/= (float scalar){
        w_ /= scalar;
        x_ /= scalar;
        y_ /= scalar;
        z_ /= scalar;
        return *this;
    }

	__device__ __host__ Quaternion_cuda operator* (const Quaternion_cuda& q) const
    {
         return Quaternion_cuda(
         w_*q.w_ - x_*q.x_ - y_*q.y_ - z_*q.z_,
         w_*q.x_ + x_*q.w_ + y_*q.z_ - z_*q.y_,
         w_*q.y_ + y_*q.w_ + z_*q.x_ - x_*q.z_,
         w_*q.z_ + z_*q.w_ + x_*q.y_ - y_*q.x_);
    }

	__device__ __host__ Quaternion_cuda operator* (float scalar) const
    {
        return Quaternion_cuda(w_ * scalar,
                       x_ * scalar,
                       y_ * scalar,
                       z_ * scalar);
    }

	__device__ __host__ Quaternion_cuda& operator*= (float scalar)
	{
		w_ *= scalar;
		x_ *= scalar;
		y_ *= scalar;
		z_ *= scalar;
		return *this;
	}

	__device__ __host__ Quaternion_cuda operator+ (const Quaternion_cuda& q) const
    {
         return Quaternion_cuda(w_ + q.w_,
                        x_ + q.x_,
                        y_ + q.y_,
                        z_ + q.z_);
    }

	__device__ __host__ Quaternion_cuda& operator+= (const Quaternion_cuda& q)
	{
		w_ += q.w_;
		x_ += q.x_;
		y_ += q.y_;
		z_ += q.z_;
		return *this;
	}

	__device__ __host__ Quaternion_cuda operator- (const Quaternion_cuda& q) const
	{
		return Quaternion_cuda(w_ - q.w_,
			x_ - q.x_,
			y_ - q.y_,
			z_ - q.z_);
	}

	__device__ __host__ Quaternion_cuda& operator-= (const Quaternion_cuda& q)
	{
		w_ -= q.w_;
		x_ -= q.x_;
		y_ -= q.y_;
		z_ -= q.z_;
		return *this;
	}

    /// Get vector part
	__device__ __host__ operator Vec3 () const{
        return Vec3(x_, y_, z_);
    }

    /// Get scalar part
	__device__ __host__ operator float() const{
        return w_;
    }
    
	float w_;
	float x_;
	float y_;
	float z_;

};

}// END Tbx NAMESPACE ==========================================================

#endif // DYNAMIC_FUSION_QUATERNION_CU_HPP
