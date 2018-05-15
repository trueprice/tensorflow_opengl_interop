#ifndef ENUMERATIONS_H
#define ENUMERATIONS_H

#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GL/gl.h>

namespace fribr
{

#define TEXTURE_FORMAT_LIST(FUNCTION) \
    FUNCTION(R32F,    CV_32FC1, GL_R32F,    GL_RED,    GL_RED,    GL_FLOAT), \
    FUNCTION(RG32F,   CV_32FC2, GL_RG32F,   GL_RG,     GL_RG,     GL_FLOAT),	\
    FUNCTION(RGB32F,  CV_32FC3, GL_RGB32F,  GL_RGB,    GL_BGR,    GL_FLOAT),	\
    FUNCTION(RGBA32F, CV_32FC4, GL_RGBA32F, GL_RGBA,   GL_BGRA,   GL_FLOAT),	\
    FUNCTION(R8,      CV_8UC1,  GL_R8,      GL_RED,    GL_RED,    GL_UNSIGNED_BYTE), \
    FUNCTION(RG8,     CV_8UC2,  GL_RG8,     GL_RG,     GL_RG,     GL_UNSIGNED_BYTE),  \
    FUNCTION(RGB8,    CV_8UC3,  GL_RGB8,    GL_RGB,    GL_BGR,    GL_UNSIGNED_BYTE),  \
    FUNCTION(RGBA8,   CV_8UC4,  GL_RGBA8,   GL_RGBA,   GL_BGRA,   GL_UNSIGNED_BYTE)

#define FRIBR_FORMAT(FRIBR, OPENCV, GL_INT_FORMAT, GL_FORMAT, GL_READ_FORMAT, GL_TYPE) FRIBR

struct TextureFormat
{
    enum { INVALID = -1, TEXTURE_FORMAT_LIST(FRIBR_FORMAT) };
    static int to_opengl_internal_format[];
    static int to_opengl_format[];
    static int to_opengl_read_format[];
    static int to_opengl_type[];
    static int to_opencv_type[];
};

// Clean up the preprocessor defines, but leave TEXTURE_FORMAT_LIST
// alive for gl_wrappers/enumerations.cpp
#undef FRIBR_FORMAT
#ifndef ENUMERATIONS_CPP
#undef TEXTURE_FORMAT_LIST 
#endif

enum ReadbackMode { ReadRGB, ReadBGR };

}

#endif
