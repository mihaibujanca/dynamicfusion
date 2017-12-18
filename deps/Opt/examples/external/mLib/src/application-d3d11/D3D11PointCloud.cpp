


void ml::D3D11PointCloud::updateColors(const std::vector<vec4f> &newValues) 
{	
	auto &vertices = m_points;

	if (newValues.size() != vertices.size()) {
		throw MLIB_EXCEPTION("vertex buffer size doesn't match");
	}
	for (size_t i = 0; i < newValues.size(); i++) {
		vertices[i].color = newValues[i];
	}
	createGPU();
}


void ml::D3D11PointCloud::releaseGPU()
{
	SAFE_RELEASE(m_vertexBuffer);
}

void ml::D3D11PointCloud::createGPU() {
	releaseGPU();
	initVB(*m_graphics);
	const std::string mLibShaderDir = util::getMLibDir() + "data/shaders/";
	m_graphics->getShaderManager().registerShader(mLibShaderDir + "defaultPointCloud.hlsl", "defaultPointCloud");
}

void ml::D3D11PointCloud::initVB(GraphicsDevice &g)
{
    if (m_points.size() == 0) return;
	auto &device = g.castD3D11().getDevice();

	size_t byteSize = sizeof(TriMeshf::Vertex) * m_points.size();
	if (byteSize > std::numeric_limits<UINT>::max()) {
		throw MLIB_EXCEPTION("buffer size too big " + std::to_string(byteSize) + ", while max is " + std::to_string(std::numeric_limits<UINT>::max()));
	}

	D3D11_BUFFER_DESC bufferDesc;
	ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
	bufferDesc.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc.ByteWidth = (UINT)byteSize;
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags = 0;

	D3D11_SUBRESOURCE_DATA data;
	ZeroMemory( &data, sizeof(data) );
    data.pSysMem = m_points.data();

	D3D_VALIDATE(device.CreateBuffer( &bufferDesc, &data, &m_vertexBuffer ));
}


void ml::D3D11PointCloud::render() const
{
    if (m_points.size() == 0) return;
	auto &context = m_graphics->getContext();

    UINT stride = sizeof(TriMeshf::Vertex);
	UINT offset = 0;
	context.IASetVertexBuffers( 0, 1, &m_vertexBuffer, &stride, &offset );
	context.IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_POINTLIST );

    context.Draw((UINT)m_points.size(), 0);
}
