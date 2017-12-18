
#ifndef APPLICATION_BASE_GRAPHICSDEVICE_H_
#define APPLICATION_BASE_GRAPHICSDEVICE_H_

namespace ml {

enum GraphicsDeviceType
{
	GraphicsDeviceTypeD3D11,
};

class D3D11GraphicsDevice;
class GraphicsDevice
{
public:
	virtual ~GraphicsDevice() {}
	virtual void init(const WindowWin32 &window) = 0;
	virtual void resize(const WindowWin32 &window) = 0;
	virtual void renderBeginFrame() = 0;
	//! clears the back buffer (color and depth)
    virtual void clear(const vec4f& clearColor = vec4f(0, 0, 0, 0), float clearDepth = 1.0f) = 0;
	virtual void renderEndFrame(bool vsync) = 0;

    void captureBackBuffer(ColorImageR8G8B8A8& result)
    {
        captureBackBufferInternal(result);
    }

	ColorImageR8G8B8A8 captureBackBuffer()
    {
		ColorImageR8G8B8A8 result;
        captureBackBuffer(result);
        return result;
    }

	GraphicsDeviceType getType() const
	{
		return m_type;
	}

	D3D11GraphicsDevice& castD3D11() const
	{
		return *((D3D11GraphicsDevice*)this);
	}

protected:
	virtual void captureBackBufferInternal(ColorImageR8G8B8A8& result) = 0;
	GraphicsDeviceType m_type;
};

}  // namespace ml

#endif  // APPLICATION_BASE_GRAPHICSDEVICE_H_
