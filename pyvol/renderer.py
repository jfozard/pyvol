
import sys
import os.path
import numpy as np
import numpy.linalg as la
import math

from mesh.GLmesh import GLMesh


import OpenGL.GL
from OpenGL.GL import (
    GL_ELEMENT_ARRAY_BUFFER,
    GL_TEXTURE_2D,
    GL_TEXTURE_3D,
    GL_TEXTURE0,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TEXTURE_WRAP_R,
    GL_CLAMP_TO_EDGE,
    GL_LINEAR,
    GL_UNPACK_ALIGNMENT,
    GL_RED,
    GL_ARRAY_BUFFER,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_TEST,
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
    glUseProgram,
    glVertexAttribPointer,
    glEnableVertexAttribArray,
    glDisableVertexAttribArray,
    glUniform1i,
    glUniform1f,
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
# The above does not work on MacOSX, so overwrite
if sys.platform == "darwin":
    from OpenGL.GL.APPLE.vertex_array_object import glGenVertexArraysAPPLE as glGenVertexArrays
    from OpenGL.GL.APPLE.vertex_array_object import glBindVertexArrayAPPLE as glBindVertexArray

from OpenGL.GL.ARB.texture_rg import (
    GL_R8,
)


from shaders.program import ShaderProgram, compile_vertex_shader_from_source, \
                            compile_fragment_shader_from_source

from transformations import Arcball, translation_matrix, scale_matrix
from imageio.tiff_parser import open_tiff

HERE = os.path.dirname(os.path.realpath(__file__))
SHADER_SOURCE_DIR = os.path.join(HERE, "shaders")


def perspective(fovy, aspect, zNear, zFar):
    f = 1.0/math.tan(fovy/2.0/180*math.pi)
    c1 = (zFar+zNear)/(zNear-zFar)
    c2 = 2*zFar*zNear/(zNear-zFar)
    return np.array(((f/aspect, 0, 0, 0),
                     (0, f, 0, 0),
                     (0, 0, c1, c2),
                     (0, 0, -1, 0)))


class StackObject(object):
    def __init__(self, fn, spacing):

        data = open_tiff(fn)

        s = np.array(data, dtype=np.uint8, order='F')

        w, h, d = s.shape
        stack_texture = glGenTextures(1)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, stack_texture)

        glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, d, h, w, 0, GL_RED,
                     GL_UNSIGNED_BYTE, s)

        self.stack_texture = stack_texture
        self.shape = s.shape


class VolumeObject(object):

    def __init__(self, fn, spacing):

        self.stack_object = StackObject(fn, spacing)

        shape = self.stack_object.shape

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        tl = np.array((shape[2]*spacing[2],
                       shape[1]*spacing[1],
                       shape[0]*spacing[0]))

        # Vertex buffer: corners of cube.
        # x, y, z, texture_x, texture_y, texture_z
        vb = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Corner 0.
              [tl[0], 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, tl[1], 0.0, 0.0, 1.0, 0.0],
              [tl[0], tl[1], 0.0, 1.0, 1.0, 0.0],
              [0.0, 0.0, tl[2], 0.0, 0.0, 1.0],
              [tl[0], 0.0, tl[2], 1.0, 0.0, 1.0],
              [0.0, tl[1], tl[2], 0.0, 1.0, 1.0],
              [tl[0], tl[1], tl[2], 1.0, 1.0, 1.0]]  # Corner 7.

        vb = np.array(vb, dtype=np.float32)
        vb = vb.flatten()

        # Triangles of cube.
        idx_out = np.array([[0, 2, 1], [2, 3, 1],  # Triangle 0, triangle 1.
                            [1, 4, 0], [1, 5, 4],
                            [3, 5, 1], [3, 7, 5],
                            [2, 7, 3], [2, 6, 7],
                            [0, 6, 2], [0, 4, 6],
                            [5, 6, 4], [5, 7, 6]],  # Triangle 10, triangle 11.
                           dtype=np.uint32)

        self.vtVBO = VBO(vb)
        self.vtVBO.bind()

        sc = 1.0/la.norm(tl)
        c = 0.5*tl

        self.transform = np.array(((0.0, 0.0, sc, -sc*c[2]),
                                   (0.0, sc, 0.0, -sc*c[1]),
                                   (sc, 0.0, 0.0, -sc*c[0]),
                                   (0.0, 0.0, 0.0, 1.0)))

        self.tex_transform = np.array( ((1.0/tl[0], 0.0, 0.0, 0.0),
                                        ( 0.0, 1.0/tl[1], 0.0, 0.0),
                                        ( 0.0, 0.0, 1.0/tl[2], 0.0),
                                        ( 0.0, 0.0, 0.0, 1.0) ))

        glBindVertexArray(0)

        self.elVBO = VBO(idx_out, target=GL_ELEMENT_ARRAY_BUFFER)
        self.elCount = len(idx_out.flatten())



class MeshObject(object):
    def __init__(self, fn, spacing):

        m = GLMesh()
        self.mesh = m
        sc = m.load_ply(fn)
        v_out, n_out, col_out, idx_out = m.generate_arrays()

        vb=np.concatenate((v_out,n_out,col_out),axis=1)
        self.elVBO=VBO(idx_out, target=GL_ELEMENT_ARRAY_BUFFER)
        self.elCount=len(idx_out.flatten())

        self.vao = glGenVertexArrays(1)


        glBindVertexArray(self.vao)

        self.vtVBO=VBO(vb)

        self.vtVBO.bind()

        glBindVertexArray(0)

        c = np.array((0,0,0))

        self.transform = np.array(((sc, 0.0, 0.0, -sc*c[0]),
                                   (0.0, sc, 0.0, -sc*c[1]),
                                   (0.0, 0.0, sc, -sc*c[2]),
                                   (0.0, 0.0, 0.0, 1.0)))


class IsosurfaceVolumeRenderer(object):

    def __init__(self):
        self.bfTex = None
        self.fbo = None
        self.volume_objects = []
        self._make_volume_shaders()

    def _make_volume_shaders(self):

        vertex = compile_vertex_shader_from_source("volume.vert")
        front_fragment = compile_fragment_shader_from_source("volume_iso_front.frag")
        back_fragment = compile_fragment_shader_from_source("volume_back.frag")

        self.b_shader = ShaderProgram(vertex, back_fragment)
        self.f_shader = ShaderProgram(vertex, front_fragment)
        self.volume_stride = 6 * 4

    def _render_volume_obj(self, volume_object, width, height, VMatrix, PMatrix):

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, width, height)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, volume_object.stack_object.stack_texture)

        glClear(GL_COLOR_BUFFER_BIT)  # Clear back buffer.

        glEnable(GL_CULL_FACE)

        glCullFace(GL_BACK)

        glUseProgram(self.b_shader.program)

        glBindVertexArray(volume_object.vao)

        volume_object.elVBO.bind()

        mv_matrix = np.dot(VMatrix, volume_object.transform)
        glUniformMatrix4fv(self.b_shader.get_uniform("mv_matrix"),
                           1, True, mv_matrix.astype('float32'))
        glUniformMatrix4fv(self.b_shader.get_uniform("p_matrix"),
                           1, True, PMatrix.astype('float32'))


        glDrawElements(GL_TRIANGLES, volume_object.elCount,
                       GL_UNSIGNED_INT, volume_object.elVBO)

        volume_object.elVBO.unbind()
        glBindVertexArray(0)
        glUseProgram(0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glActiveTexture(GL_TEXTURE0 + 1)
        glBindTexture(GL_TEXTURE_2D, self.bfTex)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, volume_object.stack_object.stack_texture)

        glUseProgram(self.f_shader.program)

        glUniform1i(self.f_shader.get_uniform("texture3s"), 0)
        glUniform1i(self.f_shader.get_uniform("backfaceTex"), 1)

        tex_inv_matrix = np.dot(PMatrix, np.dot(mv_matrix, la.inv(volume_object.tex_transform)))
        glUniformMatrix4fv(self.f_shader.get_uniform('tex_inv_matrix'), 1, True, tex_inv_matrix.astype('float32'))

        glUniform1f(self.f_shader.get_uniform('isolevel'), volume_object.threshold/255.0)


        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)

        glBindVertexArray(volume_object.vao)
        volume_object.elVBO.bind()

        glUniformMatrix4fv(self.f_shader.get_uniform("mv_matrix"),
                           1, True, mv_matrix.astype('float32'))
        glUniformMatrix4fv(self.f_shader.get_uniform("p_matrix"),
                           1, True, PMatrix.astype('float32'))

        glDrawElements(GL_TRIANGLES, volume_object.elCount,
                       GL_UNSIGNED_INT, volume_object.elVBO)

        glActiveTexture(GL_TEXTURE0+1)
        glBindTexture(GL_TEXTURE_2D, 0)

        glCullFace(GL_BACK)
        volume_object.elVBO.unbind()
        glBindVertexArray(0)
        glUseProgram(0)

    def render(self, width, height, VMatrix, PMatrix):
        for volume_object in self.volume_objects:
            self._render_volume_obj(volume_object, width, height, VMatrix, PMatrix)

    def make_volume_obj(self, fn, spacing):

        volume_object = VolumeObject(fn, spacing)
        volume_object.threshold = 15

        glBindVertexArray(volume_object.vao)

        glEnableVertexAttribArray(self.b_shader.get_attrib("position"))
        glVertexAttribPointer(self.b_shader.get_attrib("position"),
                              3,
                              GL_FLOAT,
                              False,
                              self.volume_stride,
                              volume_object.vtVBO)

        glEnableVertexAttribArray(self.b_shader.get_attrib("texcoord"))
        glVertexAttribPointer(
            self.b_shader.get_attrib("texcoord"),
            3, GL_FLOAT, False, self.volume_stride, volume_object.vtVBO+12
            )

        glBindVertexArray(0)

        self.volume_objects.append(volume_object)

    def init_back_texture(self, width, height):

        if self.fbo is None:
            self.fbo = glGenFramebuffers(1)

        glActiveTexture(GL_TEXTURE0 + 1)

        if self.bfTex is not None:
            glDeleteTextures([self.bfTex])

        self.bfTex = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.bfTex)

        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)


        w = int(width)
        h = int(height)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0,
                     GL_RGBA, GL_FLOAT, None)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        glFramebufferTexture2D(GL_FRAMEBUFFER_EXT,
                               GL_COLOR_ATTACHMENT0_EXT,
                               GL_TEXTURE_2D,
                               self.bfTex, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glBindTexture(GL_TEXTURE_2D, 0)


class SolidRenderer(object):
    def __init__(self):
        self.solid_objects = []
        self._make_solid_shaders()


    def _make_solid_shaders(self):

        vertex = compile_vertex_shader_from_source("solid_surface.vert")
        fragment = compile_fragment_shader_from_source("solid_surface.frag")

        self.shader = ShaderProgram(vertex, fragment)
        self.stride = 9 * 4


    def _render_solid_obj(self, solid_object, width, height, VMatrix, PMatrix):

        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glCullFace(GL_BACK)

        glUseProgram(self.shader.program)
        glBindVertexArray(solid_object.vao)

        solid_object.elVBO.bind()

        mv_matrix = np.dot(VMatrix, solid_object.transform)
        glUniformMatrix4fv(self.shader.get_uniform("mv_matrix"),
                           1, True, mv_matrix.astype('float32'))
        glUniformMatrix4fv(self.shader.get_uniform("p_matrix"),
                           1, True, PMatrix.astype('float32'))

        glDrawElements(GL_TRIANGLES, solid_object.elCount,
                       GL_UNSIGNED_INT, solid_object.elVBO)

        solid_object.elVBO.unbind()
        glBindVertexArray(0)
        glUseProgram(0)

    def render(self, width, height, VMatrix, PMatrix):
        for solid_object in self.solid_objects:
            self._render_solid_obj(solid_object, width, height, VMatrix, PMatrix)

    def make_solid_obj(self, fn, spacing):

        mesh_object = MeshObject(fn, spacing)


        glBindVertexArray(mesh_object.vao)



        glEnableVertexAttribArray(self.shader.get_attrib("position"))
        glVertexAttribPointer(self.shader.get_attrib("position"),
                              3,
                              GL_FLOAT,
                              False,
                              self.stride,
                              mesh_object.vtVBO)

        glEnableVertexAttribArray(self.shader.get_attrib("normal"))
        glVertexAttribPointer(
            self.shader.get_attrib("normal"),
            3, GL_FLOAT, False, self.stride, mesh_object.vtVBO+12
            )

        glEnableVertexAttribArray(self.shader.get_attrib("color"))
        glVertexAttribPointer(
            self.shader.get_attrib("color"),
            3, GL_FLOAT, False, self.stride, mesh_object.vtVBO+24
            )

        glBindVertexArray(0)

        self.solid_objects.append(mesh_object)



class VolumeRenderer(object):

    def __init__(self):
        self.bfTex = None
        self.fbo = None
        self.volume_objects = []
        self._make_volume_shaders()

    def _make_volume_shaders(self):

        vertex = compile_vertex_shader_from_source("volume.vert")
        front_fragment = compile_fragment_shader_from_source("volume_front.frag")
        back_fragment = compile_fragment_shader_from_source("volume_back.frag")

        self.b_shader = ShaderProgram(vertex, back_fragment)
        self.f_shader = ShaderProgram(vertex, front_fragment)
        self.volume_stride = 6 * 4

    def _render_volume_obj(self, volume_object, width, height, VMatrix, PMatrix):

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, width, height)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, volume_object.stack_object.stack_texture)

        glClear(GL_COLOR_BUFFER_BIT)  # Clear back buffer.

        glEnable(GL_CULL_FACE)

        glCullFace(GL_BACK)  # NB flipped

#        glValidateProgram(self.b_shader.program)
#        print("b_valid ", glGetProgramiv(self.b_shader.program,
#                                         GL_VALIDATE_STATUS))
#        print(glGetProgramInfoLog(self.b_shader.program).decode())

        glUseProgram(self.b_shader.program)

        glBindVertexArray(volume_object.vao)

        volume_object.elVBO.bind()

        mv_matrix = np.dot(VMatrix, volume_object.transform)
        glUniformMatrix4fv(self.b_shader.get_uniform("mv_matrix"),
                           1, True, mv_matrix.astype('float32'))
        glUniformMatrix4fv(self.b_shader.get_uniform("p_matrix"),
                           1, True, PMatrix.astype('float32'))

        glDrawElements(GL_TRIANGLES, volume_object.elCount,
                       GL_UNSIGNED_INT, volume_object.elVBO)

        volume_object.elVBO.unbind()
        glBindVertexArray(0)
        glUseProgram(0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glActiveTexture(GL_TEXTURE0 + 1)
        glBindTexture(GL_TEXTURE_2D, self.bfTex)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, volume_object.stack_object.stack_texture)

        glUseProgram(self.f_shader.program)

        glUniform1i(self.f_shader.get_uniform("texture3s"), 0)
        glUniform1i(self.f_shader.get_uniform("backfaceTex"), 1)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)

        glBindVertexArray(volume_object.vao)
        volume_object.elVBO.bind()

        glUniformMatrix4fv(self.f_shader.get_uniform("mv_matrix"),
                           1, True, mv_matrix.astype('float32'))
        glUniformMatrix4fv(self.f_shader.get_uniform("p_matrix"),
                           1, True, PMatrix.astype('float32'))

        glDrawElements(GL_TRIANGLES, volume_object.elCount,
                       GL_UNSIGNED_INT, volume_object.elVBO)

        glActiveTexture(GL_TEXTURE0+1)
        glBindTexture(GL_TEXTURE_2D, 0)

        glCullFace(GL_BACK)
        volume_object.elVBO.unbind()
        glBindVertexArray(0)
        glUseProgram(0)

    def render(self, width, height, VMatrix, PMatrix):
        for volume_object in self.volume_objects:
            self._render_volume_obj(volume_object, width, height, VMatrix, PMatrix)

    def make_volume_obj(self, fn, spacing):

        volume_object = VolumeObject(fn, spacing)

        glBindVertexArray(volume_object.vao)
        glEnableVertexAttribArray(self.b_shader.get_attrib("position"))
        glVertexAttribPointer(self.b_shader.get_attrib("position"),
                              3,
                              GL_FLOAT,
                              False,
                              self.volume_stride,
                              volume_object.vtVBO)

        glEnableVertexAttribArray(self.b_shader.get_attrib("texcoord"))
        glVertexAttribPointer(
            self.b_shader.get_attrib("texcoord"),
            3, GL_FLOAT, False, self.volume_stride, volume_object.vtVBO+12
            )

        glBindVertexArray(0)
        self.volume_objects.append(volume_object)

    def init_back_texture(self, width, height):

        if self.fbo is None:
            self.fbo = glGenFramebuffers(1)

        glActiveTexture(GL_TEXTURE0 + 1)

        if self.bfTex is not None:
            glDeleteTextures([self.bfTex])

        self.bfTex = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.bfTex)

        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)


        w = int(width)
        h = int(height)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0,
                     GL_RGBA, GL_FLOAT, None)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        glFramebufferTexture2D(GL_FRAMEBUFFER_EXT,
                               GL_COLOR_ATTACHMENT0_EXT,
                               GL_TEXTURE_2D,
                               self.bfTex, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glBindTexture(GL_TEXTURE_2D, 0)


class BaseWindow(object):

    def __init__(self, title, width, height):
        self.title = title
        self.width = width
        self.height = height
        self.PMatrix = np.eye(4)
        self.VMatrix = np.eye(4)
        self.initialise_window()

    def initialise_window(self):
        """Subclasses need to implement this."""
        raise(NotImplementedError())


class BaseGlutWindow(BaseWindow):

    def initialise_window(self):
        OpenGL.GLUT.glutInit([])
        OpenGL.GLUT.glutInitContextVersion(3, 2)
        OpenGL.GLUT.glutInitWindowSize(self.width, self.height)
        OpenGL.GLUT.glutInitDisplayMode(OpenGL.GLUT.GLUT_RGBA
                                        | OpenGL.GLUT.GLUT_DEPTH
                                        | OpenGL.GLUT.GLUT_DOUBLE)
        self.window = OpenGL.GLUT.glutCreateWindow("Cell surface")
        self.moving = False
        self.key_bindings = {"+": self.zoom_in,
                             "-": self.zoom_out,
                             "\x1b": self.exit}
        self.ball = Arcball()
        self.zoom = 0.5
        self.dist = 2.0

    def zoom_in(self, x=None, y=None):
        self.zoom *= 1.1
        OpenGL.GLUT.glutPostRedisplay()

    def zoom_out(self, x=None, y=None):
        self.zoom *= 0.9
        OpenGL.GLUT.glutPostRedisplay()

    def exit(self, x=None, y=None):
        sys.exit(0)

    def on_multi_button(self, bid, x, y, s):
        pass

    def on_multi_move(self, bid, x, y):
        pass

    def on_mouse_button(self, b, s, x, y):
        self.moving = not s
        self.ex, self.ey = x, y
        self.ball.down([x, y])

    def on_mouse_wheel(self, b, d, x, y):
        self.dist += self.dist/15.0 * d
        OpenGL.GLUT.glutPostRedisplay()

    def on_mouse_move(self, x, y, z=0):
        if self.moving:
            self.ex, self.ey = x, y
            self.ball.drag([x, y])
            OpenGL.GLUT.glutPostRedisplay()

    def start(self):
        self._reshape(self.width, self.height)

        OpenGL.GLUT.glutDisplayFunc(self._draw)
        OpenGL.GLUT.glutReshapeFunc(self._reshape)
        OpenGL.GLUT.glutKeyboardFunc(self.key)
        OpenGL.GLUT.glutMouseFunc(self.on_mouse_button)
        OpenGL.GLUT.glutMouseWheelFunc(self.on_mouse_button)
        OpenGL.GLUT.glutMotionFunc(self.on_mouse_move)

        OpenGL.GLUT.glutMainLoop()

    def _reshape(self, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        self.PMatrix = perspective(40.0, float(width)/height, 0.1, 10000.0)
        self.ball.place([width/2, height/2], height/2)
        self.reshape_hook()
        OpenGL.GLUT.glutPostRedisplay()

    def key(self, k, x, y):
        if k in self.key_bindings:
            func = self.key_bindings[k]
            func(x, y)

    def _draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        view_mat = translation_matrix((0, 0, -self.dist))
        view_mat = view_mat.dot(self.ball.matrix())
        view_mat = view_mat.dot(scale_matrix(self.zoom))
        self.VMatrix = view_mat
        self.draw_hook()
        OpenGL.GLUT.glutSwapBuffers()


    def draw_hook(self):
        raise(NotImplementedError())

    def reshape_hook(self):
        raise(NotImplementedError())
