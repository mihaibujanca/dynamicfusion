
namespace ml {
	
void D3D11AssetRenderer::init(GraphicsDevice &g, bool useTexture)
{
    m_constants.init(g);

    m_sphere.init(g, ml::Shapesf::sphere(1.0f, vec3f::origin));
    m_cylinder.init(g, ml::Shapesf::cylinder(0.01f, 1.0f, 2, 15, ml::vec4f(1.0f, 1.0f, 1.0f, 1.0f)));
    m_box.init(g, ml::Shapesf::box(1.0f));

    m_graphics = &g.castD3D11();

    if (useTexture)
        m_shader = &m_graphics->getShaderManager().getShaders("defaultBasicTexture");
    else
        m_shader = &m_graphics->getShaderManager().getShaders("defaultBasic");
}

void D3D11AssetRenderer::renderMesh(const D3D11TriMesh &mesh, const mat4f &cameraPerspective, const vec3f &color)
{
    renderMesh(mesh, cameraPerspective, mat4f::identity(), color);
}

void D3D11AssetRenderer::renderMesh(const D3D11TriMesh &mesh, const mat4f &cameraPerspective, const mat4f &meshToWorld, const vec3f &color)
{
    m_shader->ps.bind();
    m_shader->vs.bind();

    AssetRendererConstantBuffer constants;

    constants.worldViewProj = cameraPerspective * meshToWorld;
    constants.modelColor = ml::vec4f(color, 1.0f);
    m_constants.updateAndBind(constants, 0);
    mesh.render();
}

void D3D11AssetRenderer::renderCylinder(const mat4f &cameraPerspective, const vec3f &p0, const vec3f &p1, float radius, const vec3f &color)
{
    renderMesh(m_cylinder, cameraPerspective, mat4f::translation(p0) * mat4f::face(vec3f::eZ, p1 - p0) * mat4f::scale(1.0f, 1.0f, vec3f::dist(p0, p1)), color);
}

void D3D11AssetRenderer::renderSphere(const mat4f &cameraPerspective, const vec3f &center, float radius, const vec3f &color)
{
    renderMesh(m_sphere, cameraPerspective, mat4f::translation(center) * mat4f::scale(-radius), color);
}

void D3D11AssetRenderer::renderBox(const mat4f &cameraPerspective, const vec3f &center, float radius, const vec3f &color)
{
    renderMesh(m_box, cameraPerspective, mat4f::translation(center) * mat4f::scale(radius), color);
}

}	// namespace ml
