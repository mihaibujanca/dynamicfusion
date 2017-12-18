#include "mLibInclude.h"

#include "SimpleBuffer.h"

#include <string.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <limits>
#include <iostream>

SimpleBuffer::SimpleBuffer(std::string filename, bool onGPU, bool clampInfinity) :
    m_onGPU(onGPU)
{
    FILE* fileHandle = fopen(filename.c_str(), "rb"); //b for binary
    fread(&m_width,         sizeof(int), 1, fileHandle);
    fread(&m_height,        sizeof(int), 1, fileHandle);
    fread(&m_channelCount,  sizeof(int), 1, fileHandle);
    int datatype;
    fread(&datatype,        sizeof(int), 1, fileHandle);
    m_dataType = DataType(datatype);
    size_t elementSize = datatypeToSize(m_dataType);

    size_t size = elementSize*m_channelCount*(m_width*m_height);
    void* ptr = malloc(size);
    fread(ptr, elementSize*m_channelCount, (m_width*m_height), fileHandle);

    fclose(fileHandle);
    
    if (m_dataType == 0 && clampInfinity) {
      float* fPtr = (float*)ptr;
      for (int i = 0; i < m_width*m_height; ++i) {
	if (std::isinf(fPtr[i])) {
	  if (fPtr[i] > 0) {
	    fPtr[i] = std::numeric_limits<float>::max();
	  } else {
	    fPtr[i] = -10000.0f;
	  }
	}
      }
    }

    

    if (m_onGPU) {
        cudaMalloc(&m_data, size);
        cudaMemcpy(m_data, ptr, size, cudaMemcpyHostToDevice);
        free(ptr);
    } else {
        m_data = ptr;
    }
}

SimpleBuffer::SimpleBuffer(const SimpleBuffer& other, bool onGPU) :
    m_onGPU(onGPU),
    m_width(other.m_width),
    m_height(other.m_height),
    m_channelCount(other.m_channelCount),
    m_dataType(other.m_dataType)
{
    size_t dataSize = m_width*m_height*m_channelCount*datatypeToSize(m_dataType);
    if (onGPU) {
        cudaMalloc(&m_data, dataSize);
        if (other.m_onGPU) {
            cudaMemcpy(m_data, other.m_data, dataSize, cudaMemcpyDeviceToDevice);
        } else { 
            cudaMemcpy(m_data, other.m_data, dataSize, cudaMemcpyHostToDevice);
        }
    } else {
        m_data = malloc(dataSize);
        if (other.m_onGPU) {
            cudaMemcpy(m_data, other.m_data, dataSize, cudaMemcpyDeviceToHost);
        } else { // Both on CPU
            memcpy(m_data, other.m_data, dataSize);
        }
    }
}

void SimpleBuffer::save(std::string filename) const {
    int datatype = m_dataType;
    FILE* fileHandle = fopen(filename.c_str(), "wb"); //b for binary
    fwrite(&m_width, sizeof(int), 1, fileHandle);
    fwrite(&m_height, sizeof(int), 1, fileHandle);
    fwrite(&m_channelCount, sizeof(int), 1, fileHandle);
    fwrite(&datatype, sizeof(int), 1, fileHandle);

    size_t elementSize = datatypeToSize(m_dataType);

    size_t size = elementSize*m_channelCount*(m_width*m_height);
    void* ptr;
    if (m_onGPU) {
        ptr = malloc(size);
        cudaMemcpy(ptr, m_data, size, cudaMemcpyDeviceToHost);
    } else {
        ptr = m_data;
    }

    fwrite(ptr, elementSize*m_channelCount, (m_width*m_height), fileHandle);
    fclose(fileHandle);
    if (m_onGPU) {
        free(ptr);
    }
}

void SimpleBuffer::savePNG(std::string filenameBase, float depthScale) const {
    size_t elementSize = datatypeToSize(m_dataType);
    size_t size = elementSize*m_channelCount*(m_width*m_height);
    void* ptr;
    if (m_onGPU) {
        ptr = malloc(size);
        cudaMemcpy(ptr, m_data, size, cudaMemcpyDeviceToHost);
    }
    else {
        ptr = m_data;
    }

    std::cout << "Saving " << filenameBase << " " << m_width << "x" << m_height << "x" << m_channelCount << std::endl;
    for (int channel = 0; channel < m_channelCount; channel++)
    {
        ColorImageR8G8B8A8 image(m_width, m_height);
        for (const auto &p : image)
        {
            //elementSize*m_channelCount*(m_width*m_height)
            if (m_dataType == FLOAT)
            {
                float value = *((const float*)ptr + (p.y * m_width + p.x));
                BYTE c = util::boundToByte(value * depthScale);
                p.value = vec4uc(c, c, c, 255);
            }
        }

        LodePNG::save(image, filenameBase + std::to_string(channel) + ".png");
    }

    if (m_onGPU) {
        free(ptr);
    }
}

void SimpleBuffer::savePLYPoints(std::string filename) const {
    size_t elementSize = datatypeToSize(m_dataType);
    size_t size = elementSize*m_channelCount*(m_width*m_height);
    void* ptr;
    if (m_onGPU) {
        ptr = malloc(size);
        cudaMemcpy(ptr, m_data, size, cudaMemcpyDeviceToHost);
    }
    else {
        ptr = m_data;
    }

    std::cout << "Saving " << filename << " " << m_width << "x" << m_height << "x" << m_channelCount << std::endl;
    for (int channel = 0; channel < m_channelCount; channel++)
    {
        PointCloudf cloud;
        ColorImageR8G8B8A8 image(m_width, m_height);
        for (const auto &p : image)
        {
            if (m_dataType == FLOAT)
            {
                float value = *((const float*)ptr + (p.y * m_width + p.x));
                //std::cout << value << std::endl;
                if (value > 0.01f && value <= 10000.0f)
                {
                    cloud.m_points.push_back(vec3f((float)p.x, (float)p.y, value * 1000.0f));
                }
            }
        }
        PointCloudIOf::saveToFile(filename, cloud);
    }

    if (m_onGPU) {
        free(ptr);
    }
}

static bool isValidPixel(void* ptr, int index) {
    float value = *((const float*)ptr + index);
    return (value > 0.01f && value <= 10000.0f);
}

void SimpleBuffer::savePLYMesh(std::string filename) const {
    size_t elementSize = datatypeToSize(m_dataType);
    size_t size = elementSize*m_channelCount*(m_width*m_height);
    void* ptr;
    if (m_onGPU) {
        ptr = malloc(size);
        cudaMemcpy(ptr, m_data, size, cudaMemcpyDeviceToHost);
    }
    else {
        ptr = m_data;
    }

    std::cout << "Saving " << filename << " " << m_width << "x" << m_height << "x" << m_channelCount << std::endl;
    
    vector<vec3f> vertices;
    vector<UINT> indices;
    ColorImageR8G8B8A8 image(m_width, m_height);
    for (const auto &p : image)
    {
        int i00 = (p.y * m_width + p.x);
        float value = *((const float*)ptr + i00);
        bool valid = isValidPixel(ptr, i00);
        if (!valid) {
            // Meshlab compute bounding boxes from all vertices, even ones not in faces
            // so make the vertex be close to the origin to prevent weird behavior
            value = 0.0f;
        }
        // Always put in vertices (even if invalid)... this is due to laziness of not wanting to rewrite code
        vertices.push_back(vec3f((float)p.x, (float)p.y, value * 1000.0f));
        
        
        if (valid && p.x < image.getDimX() - 1 && p.y < image.getDimY() - 1)
        {
            int i01 = (p.y + 0) * m_width + p.x + 1;
            int i10 = (p.y + 1) * m_width + p.x + 0;
            int i11 = (p.y + 1) * m_width + p.x + 1;


            if (isValidPixel(ptr, i10) && isValidPixel(ptr, i11)) {
                indices.push_back(i00);
                indices.push_back(i10);
                indices.push_back(i11);
            }
            
            if (isValidPixel(ptr, i01) && isValidPixel(ptr, i11)) {
                indices.push_back(i00);
                indices.push_back(i11);
                indices.push_back(i01);
            }

            
        }
    }
    TriMeshf mesh(vertices, indices);
	MeshIOf::saveToPLY(filename, mesh.computeMeshData());

    if (m_onGPU) {
        free(ptr);
    }
}

SimpleBuffer::~SimpleBuffer() {
    if (m_onGPU) {
        cudaFree(m_data);
    } else {
        free(m_data);
    }
}
