
ID3DBlob* ml::D3D11Utility::CompileShader(const std::string& filename, const std::string& entryPoint, const std::string& shaderModel, const std::vector<std::pair<std::string, std::string>>& macros)
{
	static const bool s_bUsePreCompiledShaders = true;
	static bool b_CompiledShaderDirectoryCreated = false;

	const std::string path = util::getExecutablePath();
	const std::string compiledShaderDirectory(path + "/CompiledShaders/");
	if (!b_CompiledShaderDirectoryCreated) {
		//CreateDirectory(compiledShaderDirectory.c_str(), NULL);
		util::makeDirectory(compiledShaderDirectory);
		b_CompiledShaderDirectoryCreated = true;
	}
	std::string compiledFilename(compiledShaderDirectory);

	std::string fileName(filename.begin(), filename.end());
	unsigned int found = (unsigned int)fileName.find_last_of("/\\");
	fileName = fileName.substr(found + 1);
	compiledFilename.append(fileName);


	compiledFilename.push_back('.');
	unsigned int oldLen = (unsigned int)compiledFilename.length();
	compiledFilename.resize(entryPoint.length() + oldLen);
	std::copy(entryPoint.begin(), entryPoint.end(), compiledFilename.begin() + oldLen);

	D3D_SHADER_MACRO* shader_macros = nullptr;
	if (macros.size() > 0) {
		shader_macros = new D3D_SHADER_MACRO[macros.size() + 1];
		for (size_t i = 0; i < macros.size(); i++) {
			shader_macros[i].Name = macros[i].first.c_str();
			shader_macros[i].Definition = macros[i].second.c_str();
		}
		shader_macros[macros.size()].Name = NULL;
		shader_macros[macros.size()].Definition = NULL;
	}

	if (shader_macros) {
		compiledFilename.push_back('.');
		for (unsigned int i = 0; shader_macros[i].Name != NULL; i++) {
			std::string name(shader_macros[i].Name);
			if (name[0] == '\"')				name[0] = 'x';
			if (name[name.length() - 1] == '\"')	name[name.length() - 1] = 'x';
			std::string def(shader_macros[i].Definition);
			if (def[0] == '\"')				def[0] = 'x';
			if (def[def.length() - 1] == '\"')	def[def.length() - 1] = 'x';
			oldLen = (unsigned int)compiledFilename.length();
			compiledFilename.resize(oldLen + name.length() + def.length());
			std::copy(name.begin(), name.end(), compiledFilename.begin() + oldLen);
			std::copy(def.begin(), def.end(), compiledFilename.begin() + name.length() + oldLen);
		}
	}
	compiledFilename.push_back('.');
	compiledFilename.push_back('p');

	HANDLE hFindShader;
	HANDLE hFindCompiled;
	WIN32_FIND_DATAA findData_shader;
	WIN32_FIND_DATAA findData_compiled;
	hFindShader = FindFirstFileA(std::string(filename.begin(), filename.end()).c_str(), &findData_shader);
	hFindCompiled = FindFirstFileA(compiledFilename.c_str(), &findData_compiled);

	ID3DBlob* blob = nullptr;

	if (!s_bUsePreCompiledShaders || hFindCompiled == INVALID_HANDLE_VALUE || CompareFileTime(&findData_shader.ftLastWriteTime, &findData_compiled.ftLastWriteTime) > 0) {
		DWORD shaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
		// Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
		// Setting this flag improves the shader debugging experience, but still allows 
		// the shaders to be optimized and to run exactly the way they will run in 
		// the release configuration of this program.
		shaderFlags |= D3DCOMPILE_DEBUG;
#endif

		ID3DBlob* errorBlob = nullptr;
		MLIB_ASSERT_STR(util::fileExists(filename), "File not found: " + filename);
#ifdef UNICODE
		std::wstring s(filename.begin(), filename.end());
#else
		std::string s(filename.begin(), filename.end());
#endif

		//D3DX version
		HRESULT hr = D3DX11CompileFromFile(s.c_str(), shader_macros, nullptr, entryPoint.c_str(), shaderModel.c_str(), shaderFlags, 0, nullptr, &blob, &errorBlob, nullptr);

		//TODO check that this is currectly working with the shaer pre-compiler // compile-to-file	
		// D3DX is deprecated, so I've switched over to D3DCompileFromFile, which may change the dependencies you need for D3D apps.
		//HRESULT hr = D3DCompileFromFile(s.c_str(), shader_macros, nullptr, entryPoint.c_str(), shaderModel.c_str(), shaderFlags, 0, &blob, &errorBlob);
		
		if (FAILED(hr)) {
			std::string errorBlobText;
			if (errorBlob != nullptr) {
				errorBlobText = (char *)errorBlob->GetBufferPointer();
				std::cout << "Shader compilation failed for " << filename << std::endl
					<< errorBlobText << std::endl;
			}
			MLIB_ERROR("Shader compilation failed for " + filename);
		}
		if (errorBlob) errorBlob->Release();

		if (s_bUsePreCompiledShaders) {
			std::ofstream compiledFile(compiledFilename.c_str(), std::ios::out | std::ios::binary);
			compiledFile.write((char*)(blob)->GetBufferPointer(), (blob)->GetBufferSize());
			compiledFile.close();
		}
	}
	else {
		std::ifstream compiledFile(compiledFilename.c_str(), std::ios::in | std::ios::binary);
		assert(compiledFile.is_open());
		unsigned int size_data = findData_compiled.nFileSizeLow;

		HRESULT hr = D3DCreateBlob(size_data, &blob);
		if (FAILED(hr)) {
			MLIB_ERROR("Loading pre-compiled shader failed for " + filename);
		}
		else {
			compiledFile.read((char*)(blob)->GetBufferPointer(), size_data);
			compiledFile.close();
		}
	}

	if (shader_macros != nullptr) {
		SAFE_DELETE_ARRAY(shader_macros);
	}

	return blob;
}