
namespace ml {

	//template <class FloatType>
	//Camera<FloatType>::Camera(const vec3<FloatType>& eye, const vec3<FloatType>& worldUp, const vec3<FloatType>& right, FloatType fieldOfView, FloatType aspect, FloatType zNear, FloatType zFar) {
	//	m_eye = eye;
	//	m_worldUp = worldUp.getNormalized();
	//	m_right = right.getNormalized();
	//	m_look = (m_worldUp ^ m_right).getNormalized();
	//	m_up = (m_right ^ m_look).getNormalized();

	//	m_fieldOfView = fieldOfView;
	//	m_aspect = aspect;
	//	m_zNear = zNear;
	//	m_zFar = zFar;

	//	m_perspective = perspectiveFov(m_fieldOfView, m_aspect, m_zNear, m_zFar);

	//	update();
	//}

    template <class FloatType>
	Camera<FloatType>::Camera(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& worldUp, FloatType fieldOfView, FloatType aspect, FloatType zNear, FloatType zFar) {
        m_eye = eye;
        m_worldUp = worldUp.getNormalized();
        m_look = lookDir.getNormalized();
        m_right = (m_look ^ m_worldUp).getNormalized();
		m_up = (m_right ^ m_look).getNormalized();

        m_fieldOfView = fieldOfView;
        m_aspect = aspect;
        m_zNear = zNear;
        m_zFar = zFar;

        m_perspective = perspectiveFov(m_fieldOfView, m_aspect, m_zNear, m_zFar);

        update();
    }

	template <class FloatType>
	Camera<FloatType>::Camera(const Matrix4x4<FloatType>& worldToCamera, FloatType fieldOfView, FloatType aspect, FloatType zNear, FloatType zFar, bool flipRight) {
		m_eye		= vec3<FloatType>(worldToCamera(0, 3), worldToCamera(1, 3), worldToCamera(2, 3));
		m_worldUp	= vec3<FloatType>(worldToCamera(0, 1), worldToCamera(1, 1), worldToCamera(2, 1));
		m_right		= vec3<FloatType>(worldToCamera(0, 0), worldToCamera(1, 0), worldToCamera(2, 0));
		
		if (flipRight) {
			m_right = -m_right;
		}
		
		m_look = (m_worldUp ^ m_right).getNormalized();
		m_up = (m_right ^ m_look).getNormalized();

		m_fieldOfView = fieldOfView;
		m_aspect = aspect;
		m_zNear = zNear;
		m_zFar = zFar;

		m_perspective = perspectiveFov(m_fieldOfView, m_aspect, m_zNear, m_zFar);

		update();
	}

	template <class FloatType>
	Camera<FloatType>::Camera(const std::string &str)
	{
		std::istringstream s(str);
		auto read = [](std::istringstream &s, vec3<FloatType> &pt) {
			s >> pt.x >> pt.y >> pt.z;
		};
		read(s, m_eye);
		read(s, m_right);
		read(s, m_look);
		read(s, m_up);
		read(s, m_worldUp);
		s >> m_fieldOfView;
		s >> m_aspect;
		s >> m_zNear;
		s >> m_zFar;

		m_perspective = perspectiveFov(m_fieldOfView, m_aspect, m_zNear, m_zFar);

		update();
	}

	template <class FloatType>
	std::string Camera<FloatType>::toString() const
	{
		std::ostringstream s;
		auto write = [](std::ostringstream &s, const vec3<FloatType> &pt) {
			s << pt.x << ' ' << pt.y << ' ' << pt.z << ' ';
		};
		write(s, m_eye);
		write(s, m_right);
		write(s, m_look);
		write(s, m_up);
		write(s, m_worldUp);
		s << m_fieldOfView << ' ';
		s << m_aspect << ' ';
		s << m_zNear << ' ';
		s << m_zFar;
		return s.str();
	}

	template <class FloatType>
	void Camera<FloatType>::updateAspectRatio(FloatType newAspect) {
		m_aspect = newAspect;
		m_perspective = perspectiveFov(m_fieldOfView, m_aspect, m_zNear, m_zFar);
		update();
	}

	template <class FloatType>
	void Camera<FloatType>::updateFov(FloatType newFov) {
		m_fieldOfView = newFov;
		m_perspective = perspectiveFov(m_fieldOfView, m_aspect, m_zNear, m_zFar);
		update();
	}

	template <class FloatType>
	void Camera<FloatType>::update() {
		m_camera = viewMatrix(m_eye, m_look, m_up, m_right);
		m_cameraPerspective = m_perspective * m_camera;
	}

	//! angle is specified in degrees
	template <class FloatType>
	void Camera<FloatType>::lookRight(FloatType theta) {
		applyTransform(Matrix3x3<FloatType>::rotation(m_worldUp, theta));
	}

	//! angle is specified in degrees
	template <class FloatType>
	void Camera<FloatType>::lookUp(FloatType theta) {
		applyTransform(Matrix3x3<FloatType>::rotation(m_right, -theta));
	}

	//! angle is specified in degrees
	template <class FloatType>
	void Camera<FloatType>::roll(FloatType theta) {
		applyTransform(Matrix3x3<FloatType>::rotation(m_look, theta));
	}

	template <class FloatType>
	void Camera<FloatType>::applyTransform(const Matrix3x3<FloatType>& transform) {
		m_up = transform * m_up;
		m_right = transform * m_right;
		m_look = transform * m_look;
		update();
	}
	template <class FloatType>
	void Camera<FloatType>::applyTransform(const Matrix4x4<FloatType>& transform) {
		const Matrix3x3<FloatType> rot = transform.getRotation();
		m_up = rot * m_up;
		m_right = rot * m_right;
		m_look = rot * m_look;
		m_eye += transform.getTranslation();
		update();
	}

	template <class FloatType>
	void Camera<FloatType>::strafe(FloatType delta) {
		m_eye += m_right * delta;
		update();
	}

	template <class FloatType>
	void Camera<FloatType>::jump(FloatType delta) {
		m_eye += m_up * delta;
		update();
	}

	template <class FloatType>
	void Camera<FloatType>::move(FloatType delta) {
		m_eye += m_look * delta;
		update();
	}

    template <class FloatType>
    void Camera<FloatType>::translate(const vec3<FloatType> &v) {
        m_eye += v;
        update();
    }

	// resets given a new eye, lookDir, up, and right vector
	template <class FloatType>
	void Camera<FloatType>::reset(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& worldUp) {
		m_worldUp = worldUp.getNormalized();
		m_look = lookDir.getNormalized();
		m_right = (m_look ^ m_worldUp).getNormalized();
		m_up = worldUp;
		m_eye = eye;
		update();
	}

	template <class FloatType>
	Matrix4x4<FloatType> Camera<FloatType>::perspectiveFov(FloatType fieldOfView, FloatType aspectRatio, FloatType zNear, FloatType zFar) {
		FloatType width = 1.0f / tanf(math::degreesToRadians(fieldOfView) * 0.5f);
		FloatType height = aspectRatio / tanf(math::degreesToRadians(fieldOfView) * 0.5f);

		return Matrix4x4<FloatType>(
			width, 0.0f, 0.0f, 0.0f,
			0.0f, height, 0.0f, 0.0f,
			0.0f, 0.0f, zFar / (zFar - zNear), zFar * zNear / (zNear - zFar),
			0.0f, 0.0f, 1.0f, 0.0f);
	}

	template <class FloatType>
	Matrix4x4<FloatType> Camera<FloatType>::viewMatrix(const vec3<FloatType>& eye, const vec3<FloatType>& lookDir, const vec3<FloatType>& up, const vec3<FloatType>& right) {
		vec3<FloatType> l = lookDir.getNormalized();
		vec3<FloatType> u = up.getNormalized();
		vec3<FloatType> r = right.getNormalized();

		return Matrix4x4<FloatType>(r.x, r.y, r.z, -vec3<FloatType>::dot(r, eye),
			u.x, u.y, u.z, -vec3<FloatType>::dot(u, eye),
			l.x, l.y, l.z, -vec3<FloatType>::dot(l, eye),
			0.0f, 0.0f, 0.0f, 1.0f);
	}

	template <class T>
	Ray<T> Camera<T>::getScreenRay(T screenX, T screenY) const
	{
		return Rayf(m_eye, getScreenRayDirection(screenX, screenY));
	}

	template <class T>
	vec3<T> Camera<T>::getScreenRayDirection(T screenX, T screenY) const
	{
		vec3<T> perspectivePoint(math::linearMap((T)0.0, (T)1.0, (T)-1.0, (T)1.0, screenX), math::linearMap((T)0.0, (T)1.0, (T)1.0, (T)-1.0, screenY), (T)-0.5);
		return getCameraPerspective().getInverse() * perspectivePoint - m_eye;
	}

}  // namespace ml
