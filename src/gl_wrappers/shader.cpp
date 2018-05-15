#include "gl_wrappers/shader.h"

#include <vector>
#include <iostream>

// Functions from http://www.antongerdelan.net/opengl/shaders.html
// and https://www.opengl.org/wiki/Example_Code
namespace
{

void print_shader_info_log(GLuint shader_index)
{
    GLint max_length = 0;
    glGetShaderiv(shader_index, GL_INFO_LOG_LENGTH, &max_length);

    std::vector<GLchar> error_log(max_length);
    glGetShaderInfoLog(shader_index, max_length, &max_length, error_log.data());
    std::cout << "Shader " << shader_index << " compile error :" << std::endl;
    for(std::vector<GLchar>::const_iterator i = error_log.begin(); i != error_log.end(); ++i)
        std::cout << *i;
}

void print_program_info_log(GLuint program_index)
{
    GLint max_length = 0;
    glGetProgramiv(program_index, GL_INFO_LOG_LENGTH, &max_length);

    std::vector<GLchar> error_log(max_length);
    glGetProgramInfoLog(program_index, max_length, &max_length, error_log.data());
    std::cout << "Program " << program_index << " linking error :" << std::endl;
    for(std::vector<GLchar>::const_iterator i = error_log.begin(); i != error_log.end(); ++i)
        std::cout << *i;
}

bool compile(GLuint shader_index)
{
    glCompileShader(shader_index);

    // check for compile errors
    int params = -1;
    glGetShaderiv(shader_index, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params)
    {
        print_shader_info_log(shader_index);
        glDeleteShader(shader_index);
        return false;
    }

    return true;
}

}

namespace fribr
{

Shader::~Shader() throw()
{
    glDeleteShader(m_vertex_shader);
    glDeleteShader(m_fragment_shader);
    glDeleteProgram(m_program);
}

bool Shader::vertex_shader(const std::string &code)
{
    m_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    const char *c_str = code.c_str();
    glShaderSource(m_vertex_shader, 1, &c_str, NULL);
    return compile(m_vertex_shader);
}

bool Shader::fragment_shader(const std::string &code)
{
    m_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    const char *c_str = code.c_str();
    glShaderSource(m_fragment_shader, 1, &c_str, NULL);
    return compile(m_fragment_shader);
}

bool Shader::link()
{
    m_program = glCreateProgram();
    glAttachShader(m_program, m_vertex_shader);
    glAttachShader(m_program, m_fragment_shader);
    glLinkProgram(m_program);

    GLint is_linked = GL_FALSE;
    glGetProgramiv(m_program, GL_LINK_STATUS, &is_linked);
    if(is_linked == GL_FALSE)
    {
        print_program_info_log(m_program);
        glDeleteProgram(m_program);
        return false;
    }

    return true;
}

}
