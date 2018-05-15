#ifndef SHADER_H
#define SHADER_H

#include <GL/glew.h>
#include <GL/gl.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <memory>

#define GLSL(version, shader)  std::string("#version " #version "\n" #shader)
#define GLSL_RAW(shader)       std::string(#shader)

#define EXPOSE_UNIFORM_SETTER_VECTOR(TYPE, EXT) \
void set_uniform(const std::string &name, TYPE value) throw() \
{ \
    int location = glGetUniformLocation(m_program, name.c_str()); \
    glUniform ## EXT(location, 1, value.data()); \
}

#define EXPOSE_UNIFORM_SETTER_MATRIX(TYPE, EXT) \
void set_uniform(const std::string &name, TYPE value) throw() \
{ \
    int location = glGetUniformLocation(m_program, name.c_str()); \
    glUniformMatrix ## EXT(location, 1, GL_FALSE, value.data()); \
}

namespace fribr
{

class Shader
{
public:
    typedef std::shared_ptr<Shader> Ptr;
    ~Shader() throw();

    bool vertex_shader  (const std::string &code);
    bool fragment_shader(const std::string &code);
    bool link();
    void use() throw() { glUseProgram(m_program); }

    void set_uniform(const std::string &name, float value) throw()
    {
        int location = glGetUniformLocation(m_program, name.c_str());
        glUniform1f(location, value);
    }

    void set_uniform(const std::string &name, int value) throw()
    {
        int location = glGetUniformLocation(m_program, name.c_str());
        glUniform1i(location, value);
    }

    EXPOSE_UNIFORM_SETTER_MATRIX(const Eigen::Matrix2f,  2fv)
    EXPOSE_UNIFORM_SETTER_MATRIX(const Eigen::Matrix3f&, 3fv)
    EXPOSE_UNIFORM_SETTER_MATRIX(const Eigen::Matrix4f&, 4fv)
    EXPOSE_UNIFORM_SETTER_MATRIX(const Eigen::Affine3f&, 4fv)

    EXPOSE_UNIFORM_SETTER_VECTOR(const Eigen::Vector2f, 2fv)
    EXPOSE_UNIFORM_SETTER_VECTOR(const Eigen::Vector3f, 3fv)
    EXPOSE_UNIFORM_SETTER_VECTOR(const Eigen::Vector4f, 4fv)

    EXPOSE_UNIFORM_SETTER_VECTOR(const Eigen::Vector2i, 2iv)
    EXPOSE_UNIFORM_SETTER_VECTOR(const Eigen::Vector3i, 3iv)
    EXPOSE_UNIFORM_SETTER_VECTOR(const Eigen::Vector4i, 4iv)

private:
    GLuint m_vertex_shader;
    GLuint m_fragment_shader;
    GLuint m_program;

};

}

#undef EXPOSE_UNIFORM_SETTER_VECTOR
#undef EXPOSE_UNIFORM_SETTER_MATRIX

#endif // SHADER_H
