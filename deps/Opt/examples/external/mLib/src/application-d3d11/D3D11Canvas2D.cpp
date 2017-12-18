
namespace ml {

bool D3D11Canvas2D::ElementBillboard::intersects(const vec2i &mouseCoord, D3D11Canvas2D::Intersection &intersection) const
{
    if (m_box.intersects(mouseCoord))
    {
        intersection = D3D11Canvas2D::Intersection(m_id, mouseCoord - m_box.getMin());
        return true;
    }
    return false;
}

void D3D11Canvas2D::ElementBillboard::resize()
{
    bbox2f boxNdc;
    boxNdc.include(m_graphics->pixelToNDC(m_box.getMin()));
    boxNdc.include(m_graphics->pixelToNDC(m_box.getMax()));
    m_mesh.init(*m_graphics, Shapesf::rectangleZ(boxNdc.getMin(), boxNdc.getMax(), m_depth));
}

void D3D11Canvas2D::ElementBillboard::render()
{
    if (m_bUseDefaultShader) m_graphics->getShaderManager().bindShaders("defaultCanvasBillboard");
    m_tex.bind();
    m_mesh.render();
}

void D3D11Canvas2D::init(GraphicsDevice& g)
{
	releaseGPU();

	m_graphics = &g.castD3D11();
	//m_graphics->registerAsset(this);
}

D3D11Canvas2D::Intersection D3D11Canvas2D::intersectionFirst(const vec2i &mouseCoord) const
{
    D3D11Canvas2D::Intersection result;
    //
    // TODO: compute the closest intersection, instead of the first
    //
    for (auto &e : m_namedElements)
    {
        if (e.second->intersects(mouseCoord, result))
            return result;
    }
    for (Element *e : m_unnamedElements)
    {
        if (e->intersects(mouseCoord, result))
            return result;
    }
    return result;
}

void D3D11Canvas2D::releaseGPU()
{
	clearElements();
}

void D3D11Canvas2D::createGPU()
{

}

void D3D11Canvas2D::resize()
{
    for (auto &e : m_namedElements)
        e.second->resize();
    for (Element *e : m_unnamedElements)
        e->resize();
}

void D3D11Canvas2D::render()
{    
    for (auto &e : m_namedElements)
        e.second->render();
    for (Element *e : m_unnamedElements)
        e->render();
}


void D3D11Canvas2D::render(const std::string& elementId)
{
	getElementById(elementId).render();
}


}  // namespace ml
