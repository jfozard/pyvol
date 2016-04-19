
import sys
import os.path
import numpy as np
import numpy.linalg as la
import math

import OpenGL.GL
import OpenGL.GLUT
import OpenGL.GL.shaders
import OpenGL.GL.framebufferobjects

import OpenGL.arrays.vbo
from OpenGL.GL.ARB.vertex_array_object import *
from OpenGL.GL.ARB.texture_rg import *

from transformations import Arcball, translation_matrix, scale_matrix
from tiff_parser import open_tiff

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


def _compile_shader_from_source(fname, shader_type):
    """Return compiled shader; assumes fname is in shaders dir"""
    with open(os.path.join(SHADER_SOURCE_DIR, fname)) as fh:
        source = fh.read()
    return OpenGL.GL.shaders.compileShader(source, shader_type)


def compile_vertex_shader_from_source(fname):
    """Return compiled vertex shader; assumes fname is in shaders dir"""
    return _compile_shader_from_source(fname, OpenGL.GL.shaders.GL_VERTEX_SHADER)


def compile_fragment_shader_from_source(fname):
    """Return compiled fragment shader; assumes fname is in shaders dir"""
    return _compile_shader_from_source(fname, OpenGL.GL.shaders.GL_FRAGMENT_SHADER)


class ShaderProgram(object):
    """OpenGL shader program."""

    def __init__(self, vertex_shader, fragment_shader):
        program = OpenGL.GL.shaders.glCreateProgram()
        OpenGL.GL.shaders.glAttachShader(program, vertex_shader)
        OpenGL.GL.shaders.glAttachShader(program, fragment_shader)
        OpenGL.GL.shaders.glLinkProgram(program)
        # check linking error
        result = OpenGL.GL.shaders.glGetProgramiv(program, OpenGL.GL.shaders.GL_LINK_STATUS)
        if not(result):
            raise RuntimeError(glGetProgramInfoLog(program))
        self.program = program

    def get_attrib(self, name):
        return OpenGL.GL.shaders.glGetAttribLocation(self.program, name)

    def get_uniform(self, name):
        return OpenGL.GL.shaders.glGetUniformLocation(self.program, name)


class VolumeObject(object):

    def __init__(self, fn, spacing):
        self.stack_texture, shape = self.load_stack(fn)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        tl = np.array((shape[2]*spacing[2],
                       shape[1]*spacing[1],
                       shape[0]*spacing[0]))

        # Vertex buffer: corners of cube.
        # x, y, z, texture_x, texture_y, texture_z
        vb = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Corner 0.
               [ tl[0], 0.0, 0.0, 1.0, 0.0, 0.0],
               [ 0.0, tl[1], 0.0, 0.0, 1.0, 0.0],
               [ tl[0], tl[1], 0.0, 1.0, 1.0, 0.0],
               [ 0.0, 0.0, tl[2], 0.0, 0.0, 1.0],
               [ tl[0], 0.0, tl[2], 1.0, 0.0, 1.0],
               [ 0.0, tl[1], tl[2], 0.0, 1.0, 1.0],
               [ tl[0], tl[1], tl[2], 1.0, 1.0, 1.0] ]  # Corner 7.

        vb = np.array(vb, dtype=np.float32)
        vb = vb.flatten()

        # Triangles of cube.
        idx_out = np.array([[0, 2, 1], [2, 3, 1],  # Triangle 0, triangle 1.
                            [1, 4, 0], [1, 5, 4],
                            [3, 5, 1], [3, 7, 5],
                            [2, 7, 3], [2, 6, 7],
                            [0, 6, 2], [0, 4, 6],
                            [5, 6, 4], [5, 7, 6]]  # Triangle 10, triangle 11.
                            , dtype=np.uint32)
        self.vtVBO=OpenGL.arrays.vbo.VBO(vb)

        sc = 1.0/la.norm(tl)
        c = 0.5*tl

        self.transform = np.array(( (0.0, 0.0, sc, -sc*c[2]), (0.0, sc, 0.0, -sc*c[1]),  (sc, 0.0, 0.0, -sc*c[0]), (0.0, 0.0, 0.0, 1.0)))

        self.elVBO=OpenGL.arrays.vbo.VBO(idx_out, target=OpenGL.GL.GL_ELEMENT_ARRAY_BUFFER)
        self.elCount=len(idx_out.flatten())

        print('made VBO')
        self.vtVBO.bind()

    def load_stack(self, stack_fn):
        data = open_tiff(stack_fn)

        print('data shape', data.shape)

        s = np.array(data, dtype=np.uint8, order='F')

        print(s.shape)

        w, h, d = s.shape
        print('shape', s.shape)

        stack_texture = OpenGL.GL.glGenTextures(1)
        print(stack_texture)

        OpenGL.GL.glActiveTexture(OpenGL.GL.GL_TEXTURE0)
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_3D, stack_texture)

        OpenGL.GL.glTexParameter(OpenGL.GL.GL_TEXTURE_3D,
                                 OpenGL.GL.GL_TEXTURE_MAG_FILTER,
                                 OpenGL.GL.GL_LINEAR)
        OpenGL.GL.glTexParameter(OpenGL.GL.GL_TEXTURE_3D,
                                 OpenGL.GL.GL_TEXTURE_MIN_FILTER,
                                 OpenGL.GL.GL_LINEAR)

        OpenGL.GL.glPixelStorei(OpenGL.GL.GL_UNPACK_ALIGNMENT, 1)

       # glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
       # glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
       # glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        OpenGL.GL.glTexImage3D(OpenGL.GL.GL_TEXTURE_3D, 0, GL_R8, d, h, w, 0, OpenGL.GL.GL_RED, GL_UNSIGNED_BYTE, s)
        print("made 3D texture")
        return stack_texture, s.shape


class RenderWindow(object):
    def __init__(self):
        OpenGL.GLUT.glutInit([])
        OpenGL.GLUT.glutInitContextVersion(3, 2)
        OpenGL.GLUT.glutInitWindowSize(800, 600)
        OpenGL.GLUT.glutInitDisplayMode(OpenGL.GLUT.GLUT_RGBA | OpenGL.GLUT.GLUT_DEPTH)
        self.window = OpenGL.GLUT.glutCreateWindow("Cell surface")
        self.width = 800
        self.height = 600
        self.PMatrix = np.eye(4)
        self.VMatrix = np.eye(4)
        self.volue_objects = []
        self.moving = False
        self.bfTex = None
        self.fbo = None

    def initGL(self):
        self.ball = Arcball()
        self.zoom = 0.5
        self.dist = 2.0

        self.make_volume_shaders()
        self.reshape(self.width, self.height)

    def make_volume_shaders(self):

        vertex = compile_vertex_shader_from_source("volumetric.vs")
        front_fragment = compile_fragment_shader_from_source("front.frag")
        back_fragment = compile_fragment_shader_from_source("back.frag")

        self.b_shader = ShaderProgram(vertex, back_fragment)
        self.f_shader = ShaderProgram(vertex, front_fragment)
        self.volume_stride = 6 * 4


    def make_volume_obj(self, fn, spacing):

        self.volume_object = VolumeObject(fn, spacing)

        OpenGL.GL.shaders.glEnableVertexAttribArray( self.b_shader.get_attrib("position") )
        OpenGL.GL.shaders.glVertexAttribPointer(
            self.b_shader.get_attrib("position"),
            3, GL_FLOAT, False, self.volume_stride, self.volume_object.vtVBO
            )

        OpenGL.GL.shaders.glEnableVertexAttribArray( self.b_shader.get_attrib("texcoord") )
        OpenGL.GL.shaders.glVertexAttribPointer(
            self.b_shader.get_attrib("texcoord"),
            3, GL_FLOAT, False, self.volume_stride, self.volume_object.vtVBO+12
            )

        glBindVertexArray( 0 )
        OpenGL.GL.shaders.glDisableVertexAttribArray( self.b_shader.get_attrib("position") )
        OpenGL.GL.shaders.glDisableVertexAttribArray( self.b_shader.get_attrib("texcoord") )

        OpenGL.GL.glBindBuffer(OpenGL.GL.GL_ARRAY_BUFFER, 0)

        self.volue_objects.append(self.volume_object)

    def on_multi_button(self, bid, x, y, s):
        pass

    def on_multi_move(self, bid, x, y):
        pass

    def on_mouse_button(self, b, s, x, y):
        self.moving = not s
        self.ex, self.ey = x, y
        self.ball.down([x,y])

    def on_mouse_wheel(self, b, d, x, y):
        self.dist += self.dist/15.0 * d;
        OpenGL.GLUT.glutPostRedisplay()

    def on_mouse_move(self, x, y, z=0):
        if self.moving:
            self.ex, self.ey = x, y
            self.ball.drag([x,y])
            OpenGL.GLUT.glutPostRedisplay()

    def start(self):
        OpenGL.GLUT.glutDisplayFunc(self.draw)
        OpenGL.GLUT.glutReshapeFunc(self.reshape)
        OpenGL.GLUT.glutKeyboardFunc(self.key)
        OpenGL.GLUT.glutMouseFunc(self.on_mouse_button)
        OpenGL.GLUT.glutMouseWheelFunc(self.on_mouse_button)
        OpenGL.GLUT.glutMotionFunc(self.on_mouse_move)

        OpenGL.GLUT.glutMainLoop()

    def reshape(self, width, height):
        self.width = width
        self.height = height
        OpenGL.GL.glViewport(0, 0, width, height)
        self.PMatrix = perspective(40.0, float(width)/height, 0.1, 10000.0)
        self.ball.place([width/2,height/2],height/2)
        self.init_back_texture()
        OpenGL.GLUT.glutPostRedisplay()

    def draw(self):
        OpenGL.GL.glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT | OpenGL.GL.GL_DEPTH_BUFFER_BIT)
        OpenGL.GL.glClearColor(0.0,0.0,0.0,1.0)

        self.VMatrix = translation_matrix((0, 0, -self.dist)).dot(self.ball.matrix()).dot(scale_matrix(self.zoom))
        for volume_object in self.volue_objects:
            self.render_volume_obj(volume_object)
        OpenGL.GLUT.glutSwapBuffers()


    def init_back_texture(self):

        if self.fbo == None:
            self.fbo = OpenGL.GL.framebufferobjects.glGenFramebuffers(1)
        print("fbo", self.fbo)

        OpenGL.GL.glActiveTexture(OpenGL.GL.GL_TEXTURE0 + 1)

        if self.bfTex != None:
            OpenGL.GL.glDeleteTextures([self.bfTex])

        self.bfTex = OpenGL.GL.glGenTextures(1)

        print("gen Tex 1")
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D, self.bfTex)

        OpenGL.GL.glTexParameter(OpenGL.GL.GL_TEXTURE_2D,
                                 OpenGL.GL.GL_TEXTURE_MAG_FILTER,
                                 OpenGL.GL.GL_LINEAR)
        OpenGL.GL.glTexParameter(OpenGL.GL.GL_TEXTURE_2D,
                                 OpenGL.GL.GL_TEXTURE_MIN_FILTER,
                                 OpenGL.GL.GL_LINEAR)

        print("bound", self.bfTex)

        print(self.width, self.height)
        w = int(self.width)
        h = int(self.height)

        OpenGL.GL.glTexImage2D(OpenGL.GL.GL_TEXTURE_2D, 0,
                               OpenGL.GL.GL_RGBA16F, w, h, 0,
                               OpenGL.GL.GL_RGBA, GL_FLOAT, None)
        print("made texture img")

        OpenGL.GL.framebufferobjects.glBindFramebuffer(OpenGL.GL.framebufferobjects.GL_FRAMEBUFFER, self.fbo)

        OpenGL.GL.glFramebufferTexture2D(OpenGL.GL.framebufferobjects.GL_FRAMEBUFFER_EXT,
                                         OpenGL.GL.framebufferobjects.GL_COLOR_ATTACHMENT0_EXT,
                                         OpenGL.GL.GL_TEXTURE_2D,
                                         self.bfTex, 0)

        OpenGL.GL.framebufferobjects.glBindFramebuffer(OpenGL.GL.framebufferobjects.GL_FRAMEBUFFER, 0)

        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D, 0)

    def render_volume_obj(self, volume_object):

        OpenGL.GL.framebufferobjects.glBindFramebuffer(OpenGL.GL.framebufferobjects.GL_FRAMEBUFFER, self.fbo)
        OpenGL.GL.glViewport(0, 0, self.width, self.height)
        OpenGL.GL.glActiveTexture(OpenGL.GL.GL_TEXTURE0)
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_3D, volume_object.stack_texture)

        OpenGL.GL.glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT)


        OpenGL.GL.glEnable(OpenGL.GL.GL_CULL_FACE)

        OpenGL.GL.glCullFace(OpenGL.GL.GL_BACK) #NB flipped

#        glValidateProgram(self.b_shader.program)
#        print("b_valid ", glGetProgramiv(self.b_shader.program, GL_VALIDATE_STATUS))
#        print(glGetProgramInfoLog(self.b_shader.program).decode())

        OpenGL.GL.shaders.glUseProgram(self.b_shader.program)


        glBindVertexArray( volume_object.vao )
        print("copied", volume_object.elVBO.copied)
        volume_object.elVBO.bind()

        mv_matrix = np.dot(self.VMatrix, volume_object.transform)
        OpenGL.GL.shaders.glUniformMatrix4fv(self.b_shader.get_uniform("mv_matrix"), 1, True, mv_matrix.astype('float32'))
        OpenGL.GL.shaders.glUniformMatrix4fv(self.b_shader.get_uniform("p_matrix"), 1, True, self.PMatrix.astype('float32'))

        OpenGL.GL.glDrawElements(
                OpenGL.GL.GL_TRIANGLES, volume_object.elCount,
                OpenGL.GL.GL_UNSIGNED_INT, volume_object.elVBO
            )

        volume_object.elVBO.unbind()
        glBindVertexArray( 0 )
        OpenGL.GL.shaders.glUseProgram(0)

        OpenGL.GL.framebufferobjects.glBindFramebuffer(OpenGL.GL.framebufferobjects.GL_FRAMEBUFFER, 0)

        OpenGL.GL.glActiveTexture(OpenGL.GL.GL_TEXTURE0+1)
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D, self.bfTex)

        OpenGL.GL.glActiveTexture(OpenGL.GL.GL_TEXTURE0)
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_3D, volume_object.stack_texture)


        OpenGL.GL.shaders.glUseProgram(self.f_shader.program)

        OpenGL.GL.shaders.glUniform1i(self.f_shader.get_uniform("texture3s"), 0)
        OpenGL.GL.shaders.glUniform1i(self.f_shader.get_uniform("backfaceTex"), 1)


        OpenGL.GL.glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT | OpenGL.GL.GL_DEPTH_BUFFER_BIT )

        OpenGL.GL.glEnable(OpenGL.GL.GL_CULL_FACE)
        OpenGL.GL.glCullFace(OpenGL.GL.GL_FRONT)

        OpenGL.GL.glBindVertexArray(volume_object.vao)
        volume_object.elVBO.bind()

        OpenGL.GL.shaders.glUniformMatrix4fv(self.f_shader.get_uniform("mv_matrix"), 1, True, mv_matrix.astype('float32'))
        OpenGL.GL.shaders.glUniformMatrix4fv(self.f_shader.get_uniform("p_matrix"), 1, True, self.PMatrix.astype('float32'))

        OpenGL.GL.glDrawElements(
                OpenGL.GL.GL_TRIANGLES, volume_object.elCount,
                OpenGL.GL.GL_UNSIGNED_INT, volume_object.elVBO
            )

        OpenGL.GL.glActiveTexture(OpenGL.GL.GL_TEXTURE0+1)
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D, 0)

        OpenGL.GL.glCullFace(OpenGL.GL.GL_BACK)
        volume_object.elVBO.unbind()
        OpenGL.GL.glBindVertexArray( 0 )
        OpenGL.GL.shaders.glUseProgram(0)

    def key(self, k, x, y):
        if k=='+':
            self.zoom *= 1.1
        elif k=='-':
            self.zoom *= 0.9
        OpenGL.GLUT.glutPostRedisplay()


def main():
    r = RenderWindow()
    if len(sys.argv)>=5:
        spacing = map(float, sys.argv[2:5])
    else:
        spacing = (1.0, 1.0, 1.0)
    r.initGL()
    r.make_volume_obj(sys.argv[1], spacing)
    r.start()

if __name__ == '__main__':
    main()
