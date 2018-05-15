#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include "enumerations.h"
#include "texture.h"

#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GL/gl.h>

#include <Eigen/Core>

#include <vector>
#include <memory>
#include <exception>

namespace fribr {

class Framebuffer {
 public:
  typedef std::shared_ptr<Framebuffer> Ptr;

  Framebuffer(Eigen::Vector2i resolution,
              const std::vector<Texture::Descriptor> &
                  descriptors) throw(std::invalid_argument);
  ~Framebuffer() throw();

  cv::Mat read_texture(size_t index,
                       ReadbackMode mode) throw(std::out_of_range);
  const std::vector<Texture::Ptr> &get_textures() const throw() {
    return m_textures;
  }
  Eigen::Vector2i get_resolution() const throw() { return m_resolution; }

  void set_resolution(Eigen::Vector2i resolution) throw(std::invalid_argument);

  void bind() throw();
  void unbind() const throw();

 private:
  Eigen::Vector2i m_resolution;
  GLint m_old_viewport[4];
  GLuint m_fbo;
  GLint m_old_fbo;
  GLuint m_depth_buffer;
  std::vector<Texture::Descriptor> m_descriptors;
  std::vector<Texture::Ptr> m_textures;

  // Deliberately left unimplmented, Framebuffer is non-copyable.
  Framebuffer(const Framebuffer &rhs);
  Framebuffer &operator=(const Framebuffer &rhs);
};

cv::Mat read_framebuffer(Eigen::Vector2i resolution, size_t format,
                         ReadbackMode mode) throw();
}

#endif
