#include "gl_wrappers/texture.h"
#include <sstream>

namespace
{

GLint to_mag_filter(GLint min_filter)
{
    switch(min_filter)
    {
    case GL_NEAREST_MIPMAP_LINEAR:
    case GL_NEAREST_MIPMAP_NEAREST:
        return GL_NEAREST;
    case GL_LINEAR_MIPMAP_LINEAR:
    case GL_LINEAR_MIPMAP_NEAREST:
        return GL_LINEAR;
    };
    return min_filter;
}

GLuint upload_image_to_texture(const cv::Mat &image, fribr::Texture::Descriptor descriptor)
{
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, descriptor.clamp);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, descriptor.clamp);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, descriptor.filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, to_mag_filter(descriptor.filter));

    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_UNPACK_ALIGNMENT, (image.step & 3) ? 1 : 4);

    //set length of one complete row in data (doesn't need to equal image.cols)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, image.step / image.elemSize());

    // Enable anisotropic filtering.
    if ((descriptor.flags & fribr::Texture::ANISOTROPIC_FILTERING) &&
        glewGetExtension("GL_EXT_texture_filter_anisotropic"))
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0f);

    // TODO: Make this general with lookup tables.
    if (image.type() == CV_8UC1)
    {
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_R8,
                     image.cols,
                     image.rows,
                     0,
                     GL_RED,
                     GL_UNSIGNED_BYTE,
                     image.data);
    }
    else if (image.type() == CV_8UC3)
    {
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGB8,
                     image.cols,
                     image.rows,
                     0,
                     GL_BGR,
                     GL_UNSIGNED_BYTE,
                     image.data);
    }
    else if (image.type() == CV_8UC4)
    {
        cv::Mat copy(image.rows, image.cols, image.type());
        for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec4b c = image.at<cv::Vec4b>(y, x);
            copy.at<cv::Vec4b>(y, x) = cv::Vec4b(c[2], c[1], c[0], c[3]);
        }

        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGBA8,
                     copy.cols,
                     copy.rows,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     copy.data);
    }
    else if (image.type() == CV_32FC1)
    {
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_R32F,
                     image.cols,
                     image.rows,
                     0,
                     GL_RED,
                     GL_FLOAT,
                     image.data);
    }

    if (descriptor.flags & fribr::Texture::GENERATE_MIPMAPS)
        glGenerateMipmap(GL_TEXTURE_2D);

    // Restore the GL-state, otherwise this messes up other readbacks.
    glPixelStorei(GL_UNPACK_ALIGNMENT,  4);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    glBindTexture(GL_TEXTURE_2D, 0);

    return texture_id;
}

}

namespace fribr
{

Texture::Texture(const std::string& file_path, Descriptor descriptor) throw(std::invalid_argument)
    : m_texture_id(0)
{
    cv::Mat image = cv::imread(file_path);
    if (image.empty())
    {
        std::stringstream error_stream;
        error_stream << "Loading image " << file_path << " failed.";
        throw std::invalid_argument(error_stream.str());
    }
    m_resolution = Eigen::Vector2i(image.cols, image.rows);
    cv::flip(image, image, 0);
    m_texture_id = upload_image_to_texture(image, descriptor);
}

Texture::Texture(const cv::Mat &image, Descriptor descriptor) throw()
    : m_texture_id(0), m_resolution(image.cols, image.rows)
{
    cv::Mat temp_image;
    cv::flip(image, temp_image, 0);
    m_texture_id = upload_image_to_texture(temp_image, descriptor);
}

Texture::Texture(const Eigen::Vector2i resolution, Descriptor descriptor) throw()
    : m_texture_id(0), m_resolution(resolution)
{
    glGenTextures(1, &m_texture_id);
    glBindTexture(GL_TEXTURE_2D, m_texture_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, descriptor.filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, to_mag_filter(descriptor.filter));

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, descriptor.clamp);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, descriptor.clamp);

    // Enable anisotropic filtering.
    if ((descriptor.flags & Texture::ANISOTROPIC_FILTERING) &&
        glewGetExtension("GL_EXT_texture_filter_anisotropic"))
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0f);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 TextureFormat::to_opengl_internal_format[descriptor.format],
                 resolution.x(),
                 resolution.y(),
                 0,
                 TextureFormat::to_opengl_format[descriptor.format],
                 TextureFormat::to_opengl_type[descriptor.format],
                 0);

    if (descriptor.flags & Texture::GENERATE_MIPMAPS)
        glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

Texture::~Texture()
{
    if (m_texture_id)
        glDeleteTextures(1, &m_texture_id);
}

}
