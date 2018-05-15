#ifndef TEXTURE_H
#define TEXTURE_H

#include "enumerations.h"

#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GL/gl.h>

#include <Eigen/Core>

#include <string>
#include <memory>
#include <exception>

namespace fribr
{

class Texture
{
public:
    struct Descriptor
    {
        GLint  clamp;
        GLint  filter;
        size_t flags;
        int    format;
        Descriptor(GLint _clamp, GLint _filter, size_t _flags, int _format = -1)
	        : clamp(_clamp), filter(_filter), flags(_flags), format(_format)
        {
        }
    };

    enum Flags { GENERATE_MIPMAPS = 1 << 0, ANISOTROPIC_FILTERING = 1 << 1 };
    typedef std::shared_ptr<Texture> Ptr;

    Texture(const std::string &file_path,     Descriptor descriptor) throw(std::invalid_argument);
    Texture(const cv::Mat &image,             Descriptor descriptor) throw();
    Texture(const Eigen::Vector2i resolution, Descriptor descriptor) throw();
    ~Texture() throw();

    static void   unbind()                        throw() { glBindTexture(GL_TEXTURE_2D, 0);            }
           void   bind()                    const throw() { glBindTexture(GL_TEXTURE_2D, m_texture_id); }
                                      
           Eigen::Vector2i get_resolution() const throw() { return m_resolution;     }
           size_t get_height()              const throw() { return m_resolution.y(); }
           size_t get_width()               const throw() { return m_resolution.x(); }
           GLuint get_id()                  const throw() { return m_texture_id;     }
private:
    GLuint          m_texture_id;
    Eigen::Vector2i m_resolution;

    // Deliberately left unimplmented, Texture is non-copyable.
    Texture(const Texture &rhs);
    Texture& operator=(const Texture &rhs);
};

}

#undef FRIBR_FORMAT

#endif // TEXTURE_H
