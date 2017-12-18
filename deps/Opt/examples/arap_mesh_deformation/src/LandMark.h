#ifndef __LANDMARK__H__
#define __LANDMARK__H__

#include <iostream>

class LandMark
{
public:

	LandMark(std::vector<float>& position, unsigned int vertexIndex, float radius)
	{
		m_position = position;
		m_vertexIndex = vertexIndex;
		m_radius = radius;
	}

	void setPosition(std::vector<float>& position)
	{
		m_position = position;
	}

	std::vector<float> getPosition() const
	{
		return m_position;
	}

	void setVertexIndex(unsigned int vertexIndex)
	{
		m_vertexIndex = vertexIndex;
	}

	unsigned int getVertexIndex() const
	{
		return m_vertexIndex;
	}

	float getRadius() const
	{
		return m_radius;
	}

	~LandMark()
	{
	}

private:

	std::vector<float> m_position;
	unsigned int m_vertexIndex;

	float m_radius;
};

std::ostream& operator<<(std::ostream &os, const LandMark& landMark)
{
	os << landMark.getPosition()[0] << " " << landMark.getPosition()[1] << " " << landMark.getPosition()[2] << " " << landMark.getRadius() << " " << landMark.getVertexIndex();
	
	return os;
}

#endif //__LANDMARK__H__
