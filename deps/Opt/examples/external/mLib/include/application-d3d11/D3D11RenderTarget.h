
#ifndef APPLICATION_D3D11_D3D11RENDERTARGET_H_
#define APPLICATION_D3D11_D3D11RENDERTARGET_H_

namespace ml {

class D3D11RenderTarget : public GraphicsAsset
{
public:
	D3D11RenderTarget()
	{
		m_graphics = nullptr;
		m_width = 0;
		m_height = 0;

		m_targets = nullptr;
		m_targetsRTV = nullptr;
		m_targetsSRV = nullptr;

		m_depthStencil = nullptr;
		m_depthStencilDSV = nullptr;
		m_depthStencilSRV = nullptr;

		m_captureTextures = nullptr;
		m_captureDepth = nullptr;

		m_bHasSRVs = false;
	}

	D3D11RenderTarget(GraphicsDevice &g, unsigned int width, unsigned int height, const std::vector<DXGI_FORMAT>& formats = std::vector < DXGI_FORMAT > {DXGI_FORMAT_R8G8B8A8_UNORM}, bool createSRVs = false)
    {
		m_graphics = nullptr;
		m_width = 0;
		m_height = 0;

		m_targets = nullptr;
		m_targetsRTV = nullptr;
		m_targetsSRV = nullptr;

		m_depthStencil = nullptr;
		m_depthStencilDSV = nullptr;
		m_depthStencilSRV = nullptr;

		m_captureTextures = nullptr;
		m_captureDepth = nullptr;

		m_bHasSRVs = false;

        load(g, width, height, formats, createSRVs);
    }

	~D3D11RenderTarget()
	{
		releaseGPU();
	}

    // create a new render target with given width and height. Also creates an equal-sized depth buffer.
	void load(GraphicsDevice &g, unsigned int width, unsigned int height, const std::vector<DXGI_FORMAT>& formats = std::vector < DXGI_FORMAT > {DXGI_FORMAT_R8G8B8A8_UNORM}, bool createSRVs = false) 
	{
		m_graphics = &g.castD3D11();
		m_width = width;
		m_height = height;

		m_textureFormats = formats;
		m_bHasSRVs = createSRVs;

		//g.castD3D11().registerAsset(this);

		createGPU();
	}

	void releaseGPU();
	void createGPU();

    // sets the render and depth buffers as the render target for the current device.
    // to return to the original graphics device render target, call bindRenderTarget() on the graphics device.
    void bind();

	//! restores the default render target
	void unbind() {
		m_graphics->bindRenderTarget();
	}

    // clears the render and depth buffers
    void clear(const vec4f& clearColor = vec4f(0.0f), float clearDepth = 1.0f);
    void clearColor(const vec4f& clearColor = vec4f(0.0f));
	void clearDepth(float clearDepth = 1.0f);
	  
	// get the i-th color buffer; could be ColorImageR8G8B8A8 or ColorImageR32G32B32A32
	template <class T>	void captureColorBuffer(BaseImage<T>& result, unsigned int which = 0);
	void captureDepthBuffer(DepthImage32& result);										//get the raw depth buffer
	void captureDepthBuffer(DepthImage32& result, const mat4f& perspectiveTransform);	//transforms the depth back to camera space
	void captureDepthBuffer(PointImage& result, const mat4f& perspectiveTransform);		//transforms it back to 3d camera coordinate

    unsigned int getWidth() { return m_width; }
    unsigned int getHeight() { return m_height; }

	unsigned int getNumTargets() const {
		return (unsigned int)m_textureFormats.size();
	}
	bool hasSRVs() const {
		return m_bHasSRVs;
	}

	ID3D11ShaderResourceView* getColorSRV(unsigned int which = 0) {
		if (!hasSRVs()) throw MLIB_EXCEPTION("render target has no SRVs");
		return m_targetsSRV[which];
	}
	ID3D11ShaderResourceView* getDepthSRV() {
		if (!hasSRVs()) throw MLIB_EXCEPTION("render target has no SRVs");
		return m_depthStencilSRV;
	}

	void bindColorSRVs(unsigned int startSlot = 0) {
		if (!hasSRVs()) throw MLIB_EXCEPTION("render target has no SRVs");
		m_graphics->getContext().PSSetShaderResources(startSlot, getNumTargets(), m_targetsSRV);
	}

	void bindDepthSRV(unsigned int startSlot = 0) {
		if (!hasSRVs()) throw MLIB_EXCEPTION("render target has no SRVs");
		m_graphics->getContext().PSSetShaderResources(startSlot, 1, &m_depthStencilSRV);
	}

	void unbindColorSRVs(unsigned int startSlot = 0) {
		ID3D11ShaderResourceView* const srvs[] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
		m_graphics->getContext().PSSetShaderResources(startSlot, getNumTargets(), srvs);
	}

	void unbindDepthSRV(unsigned int startSlot = 0) {
		ID3D11ShaderResourceView* const srvs[] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
		m_graphics->getContext().PSSetShaderResources(startSlot, 1, srvs);
	}
private:
	D3D11GraphicsDevice *m_graphics;
	unsigned int m_width;
	unsigned int m_height;

	std::vector<DXGI_FORMAT>	m_textureFormats;

    ID3D11Texture2D**			m_targets;
	ID3D11RenderTargetView**	m_targetsRTV;
	ID3D11ShaderResourceView**	m_targetsSRV;

	ID3D11Texture2D*			m_depthStencil;
	ID3D11DepthStencilView*		m_depthStencilDSV;
	ID3D11ShaderResourceView*	m_depthStencilSRV;

    ID3D11Texture2D** m_captureTextures;    
    ID3D11Texture2D* m_captureDepth;   
    
	bool m_bHasSRVs;
};

}  // namespace ml

#endif  // APPLICATION_D3D11_D3D11RENDERTARGET_H_
