
namespace ml
{
	void D3D11RenderTarget::releaseGPU() {
		
		for (unsigned int i = 0; i < getNumTargets(); i++) {
			if (m_targets)		SAFE_RELEASE(m_targets[i]);
			if (m_targetsRTV)	SAFE_RELEASE(m_targetsRTV[i]);
			if (hasSRVs() && m_targetsSRV)	SAFE_RELEASE(m_targetsSRV[i]);
		}

		SAFE_DELETE_ARRAY(m_targets);
		SAFE_DELETE_ARRAY(m_targetsRTV);
		SAFE_DELETE_ARRAY(m_targetsSRV);

		SAFE_RELEASE(m_depthStencil);
		SAFE_RELEASE(m_depthStencilDSV);
		if (hasSRVs()) SAFE_RELEASE(m_depthStencilSRV);

		for (unsigned int i = 0; i < getNumTargets(); i++) {
			if (m_captureTextures) SAFE_RELEASE(m_captureTextures[i]);
		}
		SAFE_DELETE_ARRAY(m_captureTextures);

		SAFE_RELEASE(m_captureDepth);
	}

	void D3D11RenderTarget::createGPU() {
		releaseGPU();

		if (m_width == 0 || m_height == 0) return;

		auto &device = m_graphics->getDevice();
		auto &context = m_graphics->getContext();

		//
		// Create the render target
		//
		m_targets = new ID3D11Texture2D*[getNumTargets()];
		m_targetsRTV = new ID3D11RenderTargetView*[getNumTargets()];
		if (hasSRVs())	m_targetsSRV = new ID3D11ShaderResourceView*[getNumTargets()];
		m_captureTextures = new ID3D11Texture2D*[getNumTargets()];

		for (unsigned int i = 0; i < getNumTargets(); i++) {
			D3D11_TEXTURE2D_DESC renderDesc;
			renderDesc.Width = m_width;
			renderDesc.Height = m_height;
			renderDesc.MipLevels = 1;
			renderDesc.ArraySize = 1;
			//renderDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			renderDesc.Format = m_textureFormats[i];
			renderDesc.SampleDesc.Count = 1;
			renderDesc.SampleDesc.Quality = 0;
			renderDesc.Usage = D3D11_USAGE_DEFAULT;
			renderDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
			if (hasSRVs()) renderDesc.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
			renderDesc.CPUAccessFlags = 0;
			renderDesc.MiscFlags = 0;

			D3D_VALIDATE(device.CreateTexture2D(&renderDesc, nullptr, &m_targets[i]));				// create the texture
			D3D_VALIDATE(device.CreateRenderTargetView(m_targets[i], nullptr, &m_targetsRTV[i]));	// create the render target view
			if (hasSRVs())	D3D_VALIDATE(device.CreateShaderResourceView(m_targets[i], nullptr, &m_targetsSRV[i]));	// create the shader resource view


			// Create the color capture buffer
			renderDesc.BindFlags = 0;
			renderDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
			renderDesc.Usage = D3D11_USAGE_STAGING;
			D3D_VALIDATE(device.CreateTexture2D(&renderDesc, nullptr, &m_captureTextures[i]));
		}

		// Create the depth buffer
		D3D11_TEXTURE2D_DESC depthDesc;
		depthDesc.Width = m_width;
		depthDesc.Height = m_height;
		depthDesc.MipLevels = 1;
		depthDesc.ArraySize = 1;
		//depthDesc.Format = DXGI_FORMAT_D32_FLOAT;	//we just assume the depth buffer to always have 32 bit
		depthDesc.Format = DXGI_FORMAT_R32_TYPELESS;
		depthDesc.SampleDesc.Count = 1;
		depthDesc.SampleDesc.Quality = 0;
		depthDesc.Usage = D3D11_USAGE_DEFAULT;
		depthDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
		if (hasSRVs()) depthDesc.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
		depthDesc.CPUAccessFlags = 0;
		depthDesc.MiscFlags = 0;
		D3D_VALIDATE(device.CreateTexture2D(&depthDesc, nullptr, &m_depthStencil));

		// Create the depth view
		D3D11_DEPTH_STENCIL_VIEW_DESC depthViewDSVDesc;
		depthViewDSVDesc.Flags = 0;
		depthViewDSVDesc.Format = DXGI_FORMAT_D32_FLOAT;
		depthViewDSVDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		depthViewDSVDesc.Texture2D.MipSlice = 0;
		D3D_VALIDATE(device.CreateDepthStencilView(m_depthStencil, &depthViewDSVDesc, &m_depthStencilDSV));

		// Create the shader resource view
		D3D11_SHADER_RESOURCE_VIEW_DESC depthViewSRVDesc;
		depthViewSRVDesc.Format = DXGI_FORMAT_R32_FLOAT;
		depthViewSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		depthViewSRVDesc.Texture2D.MipLevels = 1;
		depthViewSRVDesc.Texture2D.MostDetailedMip = 0;
		if (hasSRVs())	D3D_VALIDATE(device.CreateShaderResourceView(m_depthStencil, &depthViewSRVDesc, &m_depthStencilSRV));	// create the shader resource view


		// Create the depth capture buffer
		depthDesc.BindFlags = 0;
		depthDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
		depthDesc.Usage = D3D11_USAGE_STAGING;
		D3D_VALIDATE(device.CreateTexture2D(&depthDesc, nullptr, &m_captureDepth));
	}

	void D3D11RenderTarget::bind()
	{
		MLIB_ASSERT(m_targets != nullptr);


		auto &context = m_graphics->getContext();
		context.OMSetRenderTargets(getNumTargets(), m_targetsRTV, m_depthStencilDSV);

		D3D11_VIEWPORT viewport;
		viewport.Width = (FLOAT)m_width;
		viewport.Height = (FLOAT)m_height;
		viewport.MinDepth = 0.0f;
		viewport.MaxDepth = 1.0f;
		viewport.TopLeftX = 0;
		viewport.TopLeftY = 0;
		context.RSSetViewports(1, &viewport);
	}

	void D3D11RenderTarget::clear(const vec4f& clearColor, float clearDepth)
	{
		auto &context = m_graphics->getContext();
		for (unsigned int i = 0; i < getNumTargets(); i++) {
			context.ClearRenderTargetView(m_targetsRTV[i], clearColor.array);
		}
		context.ClearDepthStencilView(m_depthStencilDSV, D3D11_CLEAR_DEPTH, clearDepth, 0);
	}

	void D3D11RenderTarget::clearColor(const vec4f& clearColor)
	{
		auto &context = m_graphics->getContext();
		for (unsigned int i = 0; i < getNumTargets(); i++) {
			context.ClearRenderTargetView(m_targetsRTV[i], clearColor.array);
		}
	}

	void D3D11RenderTarget::clearDepth(float clearDepth)
	{
		auto &context = m_graphics->getContext();
		context.ClearDepthStencilView(m_depthStencilDSV, D3D11_CLEAR_DEPTH, clearDepth, 0);
	}


	//force explicit template instantiation
	template void D3D11RenderTarget::captureColorBuffer<vec4uc>(BaseImage<vec4uc>& result, unsigned int which);
	template void D3D11RenderTarget::captureColorBuffer<vec4f>(BaseImage<vec4f>& result, unsigned int which);

	template <class T>	void D3D11RenderTarget::captureColorBuffer(BaseImage<T>& result, unsigned int which)
	{
		DXGI_FORMAT format = m_textureFormats[which];
		if (format == DXGI_FORMAT_R8G8B8A8_UNORM) {
			if (!std::is_same<vec4uc, T>::value)	throw MLIB_EXCEPTION("incompatible image format");
		}
		else if (format == DXGI_FORMAT_R32G32B32A32_FLOAT) {
			if (!std::is_same<vec4f, T>::value)		throw MLIB_EXCEPTION("incompatible image format");
		}
		else {
			throw MLIB_EXCEPTION("unknown image format");
		}

		auto &context = m_graphics->getContext();
		context.CopyResource(m_captureTextures[which], m_targets[which]);

		result.allocate(m_width, m_height);

		D3D11_MAPPED_SUBRESOURCE resource;
		UINT subresource = D3D11CalcSubresource(0, 0, 0);
		HRESULT hr = context.Map(m_captureTextures[which], subresource, D3D11_MAP_READ, 0, &resource);
		const BYTE *data = (BYTE *)resource.pData;

		for (unsigned int y = 0; y < m_height; y++)	{
			memcpy(&result(0U, y), data + resource.RowPitch * y, m_width * sizeof(T));
		}

		context.Unmap(m_captureTextures[which], subresource);
	}




	void D3D11RenderTarget::captureDepthBuffer(DepthImage32& result, const mat4f& perspectiveTransform)
	{
		captureDepthBuffer(result);

		const mat4f inv = perspectiveTransform.getInverse();

		//result.setInvalidValue(std::numeric_limits<float>::infinity());	//the default is -INF

		for (unsigned int y = 0; y < result.getHeight(); y++)	{
			for (unsigned int x = 0; x < result.getWidth(); x++)	{
				float &v = result(x, y);
				if (v >= 1.0f)
					v = result.getInvalidValue();
				else
				{
					//float dx = math::linearMap(0.0f, result.getWidth() - 1.0f, -1.0f, 1.0f, (float)x);
					//float dy = math::linearMap(0.0f, result.getHeight() - 1.0f, -1.0f, 1.0f, (float)y);
					//v = (inv * vec3f(dx, dy, v)).z;
					
					vec2f p = D3D11GraphicsDevice::pixelToNDC(vec2i(x, y), result.getWidth(), result.getHeight());
					v = (inv * vec3f(p.x, p.y, v)).z;
				}
			}
		}
	}

	void D3D11RenderTarget::captureDepthBuffer(DepthImage32 &result)
	{
		auto &context = m_graphics->getContext();
		context.CopyResource(m_captureDepth, m_depthStencil);

		result.allocate(m_width, m_height);

		D3D11_MAPPED_SUBRESOURCE resource;
		UINT subresource = D3D11CalcSubresource(0, 0, 0);
		HRESULT hr = context.Map(m_captureDepth, subresource, D3D11_MAP_READ, 0, &resource);
		const BYTE *data = (BYTE *)resource.pData;

		for (unsigned int y = 0; y < m_height; y++) {
			memcpy(&result(0U, y), data + resource.RowPitch * y, m_width * sizeof(float));
		}

		context.Unmap(m_captureDepth, subresource);
	}

}