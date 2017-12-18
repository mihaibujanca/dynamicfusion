
namespace ml {

void D3D11ShaderManager::init(GraphicsDevice &g)
{
    m_graphics = &g.castD3D11();
}

void D3D11ShaderManager::registerShader(
  const std::string& filename, 
  const std::string& shaderName, 
  const std::string& entryPointVS, 
  const std::string& shaderModelVS, 
  const std::string& entryPointPS,
  const std::string& shaderModelPS,
  const std::vector<std::pair<std::string, std::string>>& shaderMacros
  )
{
	//registerShaderWithGS(filename, shaderName, entryPointVS, entryPointVS, "", "", entryPointPS, shaderModelPS, shaderMacros);

    MLIB_ASSERT_STR(m_graphics != NULL, "shader manager not initialized");

	// in case the shader exists return
	if (m_shaders.count(shaderName) == 0) {
		auto &shaders = m_shaders[shaderName];
		shaders.vs.init(*m_graphics, filename, entryPointVS, shaderModelVS, shaderMacros);
		shaders.ps.init(*m_graphics, filename, entryPointPS, shaderModelPS, shaderMacros);
	}
}

void D3D11ShaderManager::registerShaderWithGS(
	const std::string& filename,
	const std::string& shaderName,
	const std::string& entryPointVS,
	const std::string& shaderModelVS,
	const std::string& entryPointGS,
	const std::string& shaderModelGS,
	const std::string& entryPointPS,
	const std::string& shaderModelPS,
	const std::vector<std::pair<std::string, std::string>>& shaderMacros
	)
{
	MLIB_ASSERT_STR(m_graphics != NULL, "shader manager not initialized");

	// in case the shader exists return
	if (m_shaders.count(shaderName) == 0) {
		auto &shaders = m_shaders[shaderName];
		shaders.vs.init(*m_graphics, filename, entryPointVS, shaderModelVS, shaderMacros);
		shaders.ps.init(*m_graphics, filename, entryPointPS, shaderModelPS, shaderMacros);

		if (entryPointGS != "") {
			shaders.gs.init(*m_graphics, filename, entryPointGS, shaderModelGS, shaderMacros);
		}
	}
}

void D3D11ShaderManager::bindShaders(const std::string& shaderName) const
{
	const auto& shaders = getShaders(shaderName);

	shaders.vs.bind();
	shaders.ps.bind();
	if (shaders.gs.isInit()) shaders.gs.bind();
	else m_graphics->getContext().GSSetShader(nullptr, nullptr, 0);
}


void D3D11ShaderManager::unbindShaders()
{
	m_graphics->getContext().VSSetShader(nullptr, nullptr, 0);
	m_graphics->getContext().GSSetShader(nullptr, nullptr, 0);
	m_graphics->getContext().HSSetShader(nullptr, nullptr, 0);
	m_graphics->getContext().DSSetShader(nullptr, nullptr, 0);
	m_graphics->getContext().PSSetShader(nullptr, nullptr, 0);
}


}