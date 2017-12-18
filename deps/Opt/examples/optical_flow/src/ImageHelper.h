#pragma once

#include "mLibInclude.h"

class ImageHelper
{
public:
	ImageHelper() {}
	~ImageHelper() {}


	static ml::ColorImageR8G8B8A8 convertGrayToColor(const ml::ColorImageR32& gray) {
		ml::ColorImageR8G8B8A8 res(gray.getWidth(), gray.getHeight());
		for (const auto& p : gray) {
			float v = p.value*255.0f;
			ml::vec4uc c = ml::math::round(ml::vec4f(v, v, v, 255.0f));
			res(p.x, p.y) = c;
		}
		return res;
	}

	// draw functions
	template<typename T>
	static void drawCircle(ml::BaseImage<T>& image, const ml::vec2f& center, float radius, const T& color) {
		drawCircle(image, ml::math::round(center), ml::math::round(radius), color);
	}
	template<typename T>
	static void drawCircle(ml::BaseImage<T>& image, const ml::vec2i& center, int radius, const T& color) {
		int x = radius;
		int y = 0;
		int radiusError = 1 - x;

		while (x >= y) {
			if (image.isValidCoordinate(center.x+x, center.y+y)) image(center.x+x, center.y+y) = color;
			if (image.isValidCoordinate(center.x+y, center.y+x)) image(center.x+y, center.y+x) = color;
			if (image.isValidCoordinate(center.x-x, center.y+y)) image(center.x-x, center.y+y) = color;
			if (image.isValidCoordinate(center.x-y, center.y+x)) image(center.x-y, center.y+x) = color;
			if (image.isValidCoordinate(center.x-x, center.y-y)) image(center.x-x, center.y-y) = color;
			if (image.isValidCoordinate(center.x-y, center.y-x)) image(center.x-y, center.y-x) = color;
			if (image.isValidCoordinate(center.x+x, center.y-y)) image(center.x+x, center.y-y) = color;
			if (image.isValidCoordinate(center.x+y, center.y-x)) image(center.x+y, center.y-x) = color;
			y++;
			if (radiusError < 0) {
				radiusError += 2 * y + 1;
			}
			else {
				x--;
				radiusError += 2 * (y - x) + 1;
			}
		}
	}
	template<typename T>
	static void drawLine(ml::BaseImage<T>& image, const ml::vec2i& start, const ml::vec2i& end, const T& color) {
		ml::vec2i s, e; 
		s.x = ml::math::clamp(start.x, 0, (int)image.getWidth() - 1);
		s.y = ml::math::clamp(start.y, 0, (int)image.getHeight() - 1);
		e.x = ml::math::clamp(end.x, 0, (int)image.getWidth() - 1);
		e.y = ml::math::clamp(end.y, 0, (int)image.getHeight() - 1);
		MLIB_ASSERT(image.isValidCoordinate(s.x, s.y) && image.isValidCoordinate(e.x, e.y));

		if (end.x >= start.x)
			drawLineInternal(image, s, e, color);
		else
			drawLineInternal(image, e, s, color);
	}


	static float gaussian(float sigma, float x) {
		return exp(-((x*x) / (2.0f*sigma*sigma)));
	}

	template<typename T>
	static void filterGaussian(ml::BaseImage<T>& image, float sigma) {
		const int kernelRadius = ml::math::ceil(2.0f*sigma);
		
		std::vector<float> kernel(kernelRadius+1, 0.0f);
		for (int i = 0; i < kernelRadius+1; i++) {
			kernel[i] = gaussian(sigma, (float)i);
		}

		ml::BaseImage<T> res(image.getWidth(), image.getHeight());
		for (unsigned int j = 0; j < image.getHeight(); j++) {
			for (unsigned int i = 0; i < image.getWidth(); i++) {
				T v = 0.0f;
				float w = 0.0f;
				for (int k = -kernelRadius; k <= kernelRadius; k++) {
					const int ik = i + k;
					if (ik >= 0 && ik < (int)image.getWidth()) {
						v = v + kernel[std::abs(k)]*image((unsigned int)ik, j);
						w += kernel[std::abs(k)];
					}
				}
				if (w > 0.0f)	v /= w;
				res(i, j) = v;
			}
		}
		image = res;
		for (unsigned int j = 0; j < image.getHeight(); j++) {
			for (unsigned int i = 0; i < image.getWidth(); i++) {
				T v = 0.0f;
				float w = 0.0f;
				for (int k = -kernelRadius; k <= kernelRadius; k++) {
					const int jk = j + k;
					if (jk >= 0 && jk < (int)image.getHeight()) {
						v = v + kernel[std::abs(k)] * image(i, (unsigned int)jk);
						w += kernel[std::abs(k)];
					}
				}
				if (w > 0.0f)	v /= w;
				res(i, j) = v;
			}
		}
		image = res;
	}
private:
	//! assumes end.x >= start.x
	template<typename T>
	static void drawLineInternal(ml::BaseImage<T>& image, const ml::vec2i& start, const ml::vec2i& end, const T& color) {
		//MLIB_ASSERT(image.isValidCoordinate(start.x, start.y) && image.isValidCoordinate(end.x, end.y));

		float dx = (float)(end.x - start.x);
		if (dx == 0.0f) { // vertical line
			for (int y = start.y; y < end.y; y++) image(start.x, y) = color;
			return;
		}
		float dy = (float)(end.y - start.y);
		float error = 0.0f;
		float dErr = ml::math::abs(dy / dx);
		int dir = ml::math::sign(end.y - start.y);
		int y = start.y;
		for (int x = start.x; x < end.x; x++) {
			image(x, y) = color;
			error += dErr;
			while (error >= 0.5f) {
				image(x, y) = color;
				y += dir;
				error--;
			}
		}
	}
};

