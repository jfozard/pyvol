


import sys
import os.path
import numpy as np
import numpy.linalg as la
import math

import OpenGL.GL
from OpenGL.GL import (
    GL_ELEMENT_ARRAY_BUFFER,
    GL_TEXTURE_2D,
    GL_TEXTURE_3D,
    GL_TEXTURE0,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_LINEAR,
    GL_UNPACK_ALIGNMENT,
    GL_RED,
    GL_ARRAY_BUFFER,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_RGBA,
    GL_RGBA16F,
    GL_CULL_FACE,
    GL_BACK,
    GL_FRONT,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    GL_UNSIGNED_BYTE,
    GL_FLOAT,
    glGenTextures,
    glTexImage2D,
    glTexImage3D,
    glTexParameter,
    glActiveTexture,
    glBindTexture,
    glBindBuffer,
    glPixelStorei,
    glViewport,
    glClear,
    glClearColor,
    glDeleteTextures,
    glFramebufferTexture2D,
    glEnable,
    glCullFace,
    glDrawElements,
)
import OpenGL.GLUT
from OpenGL.GL.shaders import (
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    GL_LINK_STATUS,
    compileShader,
    glCreateProgram,
    glUseProgram,
    glAttachShader,
    glLinkProgram,
    glGetProgramiv,
    glGetProgramInfoLog,
    glGetAttribLocation,
    glGetUniformLocation,
    glVertexAttribPointer,
    glEnableVertexAttribArray,
    glDisableVertexAttribArray,
    glUniform1i,
    glUniformMatrix4fv,
)
from OpenGL.GL.framebufferobjects import (
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_EXT,
    GL_COLOR_ATTACHMENT0_EXT,
    glGenFramebuffers,
    glBindFramebuffer,
)

from OpenGL.arrays.vbo import VBO
from OpenGL.GL.ARB.vertex_array_object import (
    glGenVertexArrays,
    glBindVertexArray,
)

HERE = os.path.dirname(os.path.realpath(__file__))
SHADER_SOURCE_DIR = HERE #os.path.join(HERE, "shaders")



def _compile_shader_from_source(fname, shader_type):
    """Return compiled shader; assumes fname is in shaders dir"""
    with open(os.path.join(SHADER_SOURCE_DIR, fname)) as fh:
        source = fh.read()
    return compileShader(source, shader_type)


def compile_vertex_shader_from_source(fname):
    """Return compiled vertex shader; assumes fname is in shaders dir"""
    return _compile_shader_from_source(fname,
                                       GL_VERTEX_SHADER)


def compile_fragment_shader_from_source(fname):
    """Return compiled fragment shader; assumes fname is in shaders dir"""
    return _compile_shader_from_source(fname,
                                       GL_FRAGMENT_SHADER)


class ShaderProgram(object):
    """OpenGL shader program."""

    def __init__(self, vertex_shader, fragment_shader):
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        # check linking error
        result = glGetProgramiv(program, GL_LINK_STATUS)
        if not(result):
            raise RuntimeError(glGetProgramInfoLog(program))
        self.program = program

    def get_attrib(self, name):
        return glGetAttribLocation(self.program, name)

    def get_uniform(self, name):
        return glGetUniformLocation(self.program, name)
