
void ml::D3D11PixelShader::init(
	GraphicsDevice &g, 
	const std::string &filename, 
	const std::string& entryPoint, 
	const std::string& shaderModel,
	const std::vector<std::pair<std::string, std::string>>& shaderMacros)
{
    m_graphics = &g.castD3D11();

	releaseGPU();
	SAFE_RELEASE(m_blob);

	m_filename = filename;
	//g.castD3D11().registerAsset(this);

	m_blob = D3D11Utility::CompileShader(m_filename, entryPoint, shaderModel, shaderMacros);
	MLIB_ASSERT_STR(m_blob != nullptr, "CompileShader failed");

	createGPU();
}

void ml::D3D11PixelShader::releaseGPU()
{
	SAFE_RELEASE(m_shader);
}

void ml::D3D11PixelShader::createGPU()
{
	releaseGPU();

	auto &device = m_graphics->getDevice();

	D3D_VALIDATE(device.CreatePixelShader(m_blob->GetBufferPointer(), m_blob->GetBufferSize(), nullptr, &m_shader));
}

void ml::D3D11PixelShader::bind() const
{
	m_graphics->getContext().PSSetShader(m_shader, nullptr, 0);
}