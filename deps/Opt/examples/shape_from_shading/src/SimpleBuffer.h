#ifndef SimpleBuffer_h
#define SimpleBuffer_h
#include <string>
#include <stdlib.h>
class SimpleBuffer {
public:
    enum DataType { FLOAT = 0, UCHAR = 1 };
    static size_t datatypeToSize(SimpleBuffer::DataType dt) {
        return (dt == DataType::FLOAT) ? sizeof(float) : sizeof(unsigned char);
    }
protected:
    bool        m_onGPU;
    int         m_width;
    int         m_height;
    int         m_channelCount;
    DataType    m_dataType;
    void*       m_data;

public:
    SimpleBuffer(std::string filename, bool onGPU, bool clampInfinity = true);
    SimpleBuffer(const SimpleBuffer& other, bool onGPU);

    int width() const {
        return m_width;
    }
    int height() const {
        return m_height;
    }
    void* data() const {
        return m_data;
    }

    DataType type() const {
        return m_dataType;
    }

    void save(std::string filename) const;
    void savePNG(std::string filenameBase, float depthScale) const;
    void savePLYPoints(std::string filename) const;
    void savePLYMesh(std::string filename) const;
    ~SimpleBuffer();
};

#endif
