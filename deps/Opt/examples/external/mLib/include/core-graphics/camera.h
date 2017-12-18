
#ifndef CORE_GRAPHICS_CAMERA_H_
#define CORE_GRAPHICS_CAMERA_H_

namespace ml {

	template <class FloatType>
	class Camera : public BinaryDataSerialize < Camera<FloatType> > {
	public:
		Camera() {}
		Camera(const std::string &s);

		//! Standard constructor (eye point, look direction, up vector)
		Camera(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& worldUp, FloatType fieldOfView, FloatType aspect, FloatType zNear, FloatType zFar);

		//! Construct camera from extrinsic matrix m (columns are x, y, z vectors and origin of camera in that order).
		//! If flipRight is set, flip the x coordinate
		Camera(const Matrix4x4<FloatType>& m, const FloatType fieldOfView, const FloatType aspect, const FloatType zNear, const FloatType zFar, bool flipRight = false);

		virtual void updateAspectRatio(FloatType newAspect);
		void updateFov(FloatType newFov);
		void lookRight(FloatType theta);
		void lookUp(FloatType theta);
		void roll(FloatType theta);

		void strafe(FloatType delta);
		void jump(FloatType delta);
		void move(FloatType delta);
		void translate(const vec3<FloatType> &v);

		Ray<FloatType> getScreenRay(FloatType screenX, FloatType screenY) const;
		vec3<FloatType> getScreenRayDirection(FloatType screenX, FloatType screenY) const;

		Matrix4x4<FloatType> getCamera() const {
			return m_camera;
		}

		Matrix4x4<FloatType> getPerspective() const {
			return m_perspective;
		}

		Matrix4x4<FloatType> getCameraPerspective() const {
			return m_cameraPerspective;
		}

		vec3<FloatType> getEye() const {
			return m_eye;
		}

		vec3<FloatType> getLook() const {
			return m_look;
		}

		vec3<FloatType> getRight() const {
			return m_right;
		}

		vec3<FloatType> getUp() const {
			return m_up;
		}

		FloatType getFoV() const {
			return m_fieldOfView;
		}

		FloatType getAspect() const {
			return m_aspect;
		}

		std::string toString() const;

		void applyTransform(const Matrix3x3<FloatType>& transform);
		void applyTransform(const Matrix4x4<FloatType>& transform);
		void reset(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& up);


		static Camera<FloatType> visionToGraphics(const Matrix4x4<FloatType>& extrinsic, unsigned int width, unsigned int height, FloatType fx, FloatType fy, FloatType zNear, FloatType zFar) {
			//not entirely sure whether there is a '-1' for width/height somewhere
			FloatType fov = (FloatType)2.0 * atan((FloatType)width / ((FloatType)2 * fx));
			FloatType aspect = (FloatType)width / (FloatType)height;

			return Camera<FloatType>(extrinsic.getTranspose(), math::radiansToDegrees(fov), aspect, zNear, zFar);
		}

		static Matrix4x4<FloatType> visionToGraphicsProj(unsigned int width, unsigned int height, FloatType fx, FloatType fy, FloatType zNear, FloatType zFar)
		{
			//not entirely sure whether there is a '-1' for width/height somewhere
			FloatType fov = (FloatType)2.0 * atan((FloatType)width / ((FloatType)2 * fx));
			FloatType aspect = (FloatType)width / (FloatType)height;
			return perspectiveFov(math::radiansToDegrees(fov), aspect, zNear, zFar);
		}

		//! note: this assumes the DX11/OGl structure of m (see NDC space)
		static Matrix4x4<FloatType> graphicsToVisionProj(const Matrix4x4<FloatType>& m, unsigned int width, unsigned int height)
		{
			FloatType fov = (FloatType)2.0 * atan((FloatType)1 / m(0, 0));
			FloatType aspect = (FloatType)width / height;
			FloatType t = tan((FloatType)0.5 * fov);
			FloatType focalLengthX = (FloatType)0.5 * (FloatType)width / t;
			FloatType focalLengthY = (FloatType)0.5 * (FloatType)height / t * aspect;

			focalLengthY = -focalLengthY;

			return  Matrix4x4<FloatType>(
				focalLengthX, 0.0f, (FloatType)(width - 1) / 2.0f, 0.0f,
				0.0f, focalLengthY, (FloatType)(height - 1) / 2.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f);
		}


		float getNearPlane() const {
			return m_zNear;
		}

		float getFarPlane() const {
			return m_zFar;
		}

		//! field of view is in degrees
		static Matrix4x4<FloatType> perspectiveFov(FloatType fieldOfView, FloatType aspectRatio, FloatType zNear, FloatType zFar);
		static Matrix4x4<FloatType> viewMatrix(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& up, const vec3<FloatType>& right);

	private:
		void update();

		vec3<FloatType> m_eye, m_right, m_look, m_up;
		vec3<FloatType> m_worldUp;
		Matrix4x4<FloatType> m_camera;
		Matrix4x4<FloatType> m_perspective;
		Matrix4x4<FloatType> m_cameraPerspective;

		FloatType m_fieldOfView, m_aspect, m_zNear, m_zFar;
	};

	typedef Camera<float> Cameraf;
	typedef Camera<double> Camerad;

}  // namespace ml

#include "camera.inl"

#endif  // CORE_GRAPHICS_CAMERA_H_
