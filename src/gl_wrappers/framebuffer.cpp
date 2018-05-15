#include "gl_wrappers/framebuffer.h"
#include "GL/glu.h"
#include <sstream>

namespace fribr
{

cv::Mat read_framebuffer(Eigen::Vector2i resolution, size_t format, ReadbackMode mode) throw()
{
    cv::Mat image(resolution.y(), resolution.x(), TextureFormat::to_opencv_type[format]);

    glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
    glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
    glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);

    GLint read_format = mode == ReadRGB ? TextureFormat::to_opengl_format[format]
                                        : TextureFormat::to_opengl_read_format[format];

    // Use fast 4-byte alignment (default anyway) if possible.
    glPixelStorei(GL_PACK_ALIGNMENT, (image.step & 3) ? 1 : 4);
    // Set the length of one complete row in destination data (doesn't need to equal image.cols).
    glPixelStorei(GL_PACK_ROW_LENGTH, image.step / image.elemSize());
    glReadPixels(0, 0, image.cols, image.rows,
                 read_format,
                 TextureFormat::to_opengl_type[format],
                 image.data);

    // Restore the GL-state, otherwise this messes up other readbacks.
    glPixelStorei(GL_PACK_ALIGNMENT,  4);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);

    cv::flip(image, image, 0);


    return image;
}

Framebuffer::Framebuffer(Eigen::Vector2i resolution, const std::vector<Texture::Descriptor> &descriptors) throw(std::invalid_argument)
    : m_resolution(0, 0), m_fbo(0), m_old_fbo(0), m_depth_buffer(0), m_descriptors(descriptors)
{
    set_resolution(resolution);
}

Framebuffer::~Framebuffer() throw()
{
    if (m_fbo)
        glDeleteFramebuffers(1, &m_fbo);
    if (m_depth_buffer)
        glDeleteRenderbuffers(1, &m_depth_buffer);
}

cv::Mat Framebuffer::read_texture(size_t index, ReadbackMode mode) throw(std::out_of_range)
{
    if (index > m_textures.size())
    {
        std::stringstream error_stream;
        error_stream << "Texture index out of range: " << index
                     << " not in [0, " << m_textures.size() << "]";
        throw std::out_of_range(error_stream.str());
    }

    GLint bound_fbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &bound_fbo);
    bool fbo_already_bound = bound_fbo == (int)m_fbo;
    if (!fbo_already_bound)
        bind();

    glReadBuffer(GL_COLOR_ATTACHMENT0 + index);
    int     format = m_descriptors[index].format;
    cv::Mat image  = read_framebuffer(m_resolution, format, mode);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    if (!fbo_already_bound)
        unbind();

    return image;
}
 
void Framebuffer::set_resolution(Eigen::Vector2i resolution) throw(std::invalid_argument)
{
    if (m_resolution == resolution)
        return;
    
    if (m_fbo)
        glDeleteFramebuffers(1, &m_fbo);
    if (m_depth_buffer)
        glDeleteRenderbuffers(1, &m_depth_buffer);
    m_textures.clear();

    GLint old_fbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &old_fbo);

    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    glGenRenderbuffers(1, &m_depth_buffer);
    glBindRenderbuffer(GL_RENDERBUFFER, m_depth_buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, resolution.x(), resolution.y());
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_buffer);

    for (size_t i = 0; i < m_descriptors.size(); ++i)
    {
        Texture::Descriptor texture_descriptor = m_descriptors[i];
        Texture::Ptr        texture(new Texture(resolution, texture_descriptor));

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, texture->get_id(), 0);
        m_textures.push_back(texture);
    }

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        std::stringstream error_stream;
        error_stream << "Framebuffer status not complete: " << gluErrorString(status);
        throw std::invalid_argument(error_stream.str());
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, old_fbo);
    m_resolution = resolution;    
}

void Framebuffer::bind() throw()
{
    glGetIntegerv(GL_VIEWPORT, m_old_viewport);
    glViewport(0, 0, m_resolution.x(), m_resolution.y());

    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &m_old_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    std::vector<GLenum> buffers;
    buffers.reserve(m_textures.size());
    for (size_t i = 0; i < m_textures.size(); ++i)
        buffers.push_back(GL_COLOR_ATTACHMENT0 + i);
    glDrawBuffers(buffers.size(), buffers.data());
}

void Framebuffer::unbind() const throw()
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_old_fbo);
    glViewport(m_old_viewport[0], m_old_viewport[1],
               m_old_viewport[2], m_old_viewport[3]);
}


}
