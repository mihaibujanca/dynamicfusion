
void ml::D3D11VertexShader::init(
	GraphicsDevice &g,
	const std::string& filename,
	const std::string& entryPoint,
	const std::string& shaderModel,
	const std::vector<std::pair<std::string, std::string>>& shaderMacros)
{
	m_graphics = &g.castD3D11();
	if (!util::fileExists(filename))
	{
		std::cout << "file not found: " << filename << std::endl;
		return;
	}
	releaseGPU();
	SAFE_RELEASE(m_blob);

	m_filename = filename;
	//g.castD3D11().registerAsset(this);

	m_blob = D3D11Utility::CompileShader(m_filename, entryPoint, shaderModel, shaderMacros);
	if (m_blob == nullptr) throw MLIB_EXCEPTION("CompileShader failed");

	createGPU();
}

void ml::D3D11VertexShader::releaseGPU()
{
	SAFE_RELEASE(m_shader);
	SAFE_RELEASE(m_standardLayout);
}

void ml::D3D11VertexShader::createGPU()
{
	releaseGPU();

	auto &device = m_graphics->getDevice();

	D3D_VALIDATE(device.CreateVertexShader(m_blob->GetBufferPointer(), m_blob->GetBufferSize(), nullptr, &m_shader));

	D3D_VALIDATE(device.CreateInputLayout(D3D11TriMesh::layout, D3D11TriMesh::layoutElementCount, m_blob->GetBufferPointer(), m_blob->GetBufferSize(), &m_standardLayout));
}

void ml::D3D11VertexShader::bind() const
{
	auto &context = m_graphics->getContext();
	context.VSSetShader(m_shader, nullptr, 0);
	context.IASetInputLayout(m_standardLayout);
}