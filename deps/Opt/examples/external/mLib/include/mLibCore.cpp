
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef LINUX
#ifndef _POSIX_SOURCE
#define _POSIX_SOURCE
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/time.h>
#include <dirent.h>
#endif

//
// core-base source files
//
#include "../src/core-base/common.cpp"

//
// core-math source files
//
#include "../src/core-math/rng.cpp"
#include "../src/core-math/triangleIntersection.cpp"

//
// core-util source files
//
#include "../src/core-util/utility.cpp"
#include "../src/core-util/windowsUtil.cpp"
#include "../src/core-util/directory.cpp"
#include "../src/core-util/timer.cpp"
#include "../src/core-util/pipe.cpp"
#include "../src/core-util/UIConnection.cpp"
#include "../src/core-util/eventMap.cpp"

//
// core-multithreading source files
//
#include "../src/core-multithreading/threadPool.cpp"
#include "../src/core-multithreading/workerThread.cpp"

//
// core-graphics source files
//
#include "../src/core-graphics/RGBColor.cpp"

//
// core-mesh source files
//
#include "../src/core-mesh/meshUtil.cpp"

#ifdef LINUX
namespace ml
{
    template<> const vec3f vec3f::origin(0.0f, 0.0f, 0.0f);
    template<> const vec3f vec3f::eX(1.0f, 0.0f, 0.0f);
    template<> const vec3f vec3f::eY(0.0f, 1.0f, 0.0f);
    template<> const vec3f vec3f::eZ(0.0f, 0.0f, 1.0f);

    template<> const vec3d vec3d::origin(0.0, 0.0, 0.0);
    template<> const vec3d vec3d::eX(1.0, 0.0, 0.0);
    template<> const vec3d vec3d::eY(0.0, 1.0, 0.0);
    template<> const vec3d vec3d::eZ(0.0, 0.0, 1.0);
    template<> const vec6d vec6d::origin(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    template<> const vec6f vec6f::origin(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    template<> const vec4f vec4f::origin(0.0f, 0.0f, 0.0f, 0.0f);
    template<> const vec4f vec4f::eX(1.0f, 0.0f, 0.0f, 0.0f);
    template<> const vec4f vec4f::eY(0.0f, 1.0f, 0.0f, 0.0f);
    template<> const vec4f vec4f::eZ(0.0f, 0.0f, 1.0f, 0.0f);
    template<> const vec4f vec4f::eW(0.0f, 0.0f, 0.0f, 1.0f);

    template<> const vec4d vec4d::origin(0.0, 0.0, 0.0, 0.0);
    template<> const vec4d vec4d::eX(1.0, 0.0, 0.0, 0.0);
    template<> const vec4d vec4d::eY(0.0, 1.0, 0.0, 0.0);
    template<> const vec4d vec4d::eZ(0.0, 0.0, 1.0, 0.0);
    template<> const vec4d vec4d::eW(0.0, 0.0, 0.0, 1.0);

    template<> const vec2f vec2f::origin(0.0f, 0.0f);
    template<> const vec2f vec2f::eX(1.0f, 0.0f);
    template<> const vec2f vec2f::eY(0.0f, 1.0f);

    template<> const vec2d vec2d::origin(0.0, 0.0);
    template<> const vec2d vec2d::eX(1.0, 0.0);
    template<> const vec2d vec2d::eY(0.0, 1.0);

    template<> const vec1f vec1f::origin(0.0f);
    template<> const vec1f vec1f::eX(1.0f);

    template<> const vec1d vec1d::origin(0.0);
    template<> const vec1d vec1d::eX(1.0);
}
#endif