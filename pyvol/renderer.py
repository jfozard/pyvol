
import sys
import os.path
import numpy as np
import numpy.linalg as la
import math

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
from OpenGL.GL.framebufferobjects import *

from OpenGL.arrays.vbo import *
from OpenGL.arrays import GLvoidpArray, GLintArray
from OpenGL.GL.ARB.vertex_array_object import *
from OpenGL.GL.ARB.texture_rg import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GL.shaders import *

from transformations import Arcball, translation_matrix, scale_matrix
from tiff_parser import open_tiff

HERE = os.path.dirname(os.path.realpath(__file__))
SHADER_SOURCE_DIR = os.path.join(HERE, "shaders")


def perspective(fovy, aspect, zNear, zFar):
    f = 1.0/math.tan(fovy/2.0/180*math.pi)
    return np.array(((f/aspect, 0, 0, 0), (0,f,0,0), (0,0,(zFar+zNear)/(zNear-zFar), 2*zFar*zNear/(zNear-zFar)), (0, 0, -1, 0)))


def _compile_shader_from_source(fname, shader_type):
    """Return compiled shader; assumes fname is in shaders dir"""
    with open(os.path.join(SHADER_SOURCE_DIR, fname)) as fh:
        source = fh.read()
    return compileShader(source, shader_type)


def compile_vertex_shader_from_source(fname):
    """Return compiled vertex shader; assumes fname is in shaders dir"""
    return _compile_shader_from_source(fname, GL_VERTEX_SHADER)


def compile_fragment_shader_from_source(fname):
    """Return compiled fragment shader; assumes fname is in shaders dir"""
    return _compile_shader_from_source(fname, GL_FRAGMENT_SHADER)


class Obj():
    pass


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


class RenderWindow(object):
    def __init__(self):
        glutInit([])
        glutInitContextVersion(3, 2)
        glutInitWindowSize(800, 600)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH)
        self.window = glutCreateWindow("Cell surface")
        self.width = 800
        self.height = 600
        self.PMatrix = np.eye(4)
        self.VMatrix = np.eye(4)
        self.objs = []
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

        vs = Obj()
        self.b_shader = ShaderProgram(vertex, back_fragment)

        self.volume_stride = 6 * 4

        self.f_shader = ShaderProgram(vertex, front_fragment)
        vs.f_shader = self.f_shader.program

        self.volume_shaders = vs

    def make_volume_obj(self, fn, spacing):
        o = Obj()

        o.stack_texture, shape = self.load_stack(fn)

        o.vao = glGenVertexArrays(1)
        glBindVertexArray(o.vao)
        vs = self.volume_shaders

        tl = np.array((shape[2]*spacing[2],
                       shape[1]*spacing[1],
                       shape[0]*spacing[0]))

        vb = [ [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [ tl[0], 0.0, 0.0, 1.0, 0.0, 0.0],
               [ 0.0, tl[1], 0.0, 0.0, 1.0, 0.0],
               [ tl[0], tl[1], 0.0, 1.0, 1.0, 0.0],
               [ 0.0, 0.0, tl[2], 0.0, 0.0, 1.0],
               [ tl[0], 0.0, tl[2], 1.0, 0.0, 1.0],
               [ 0.0, tl[1], tl[2], 0.0, 1.0, 1.0],
               [ tl[0], tl[1], tl[2], 1.0, 1.0, 1.0] ]

        vb = np.array(vb, dtype=np.float32)
        vb = vb.flatten()

        idx_out = np.array([[0, 2, 1], [2, 3, 1],
                            [1, 4, 0], [1, 5, 4],
                            [3, 5, 1], [3, 7, 5],
                            [2, 7, 3], [2, 6, 7],
                            [0, 6, 2], [0, 4, 6],
                            [5, 6, 4], [5, 7, 6]]
                            , dtype=np.uint32)
        o.vtVBO=VBO(vb)

        print('made VBO')
        o.vtVBO.bind()

        glEnableVertexAttribArray( self.b_shader.get_attrib("position") )
        glVertexAttribPointer(
            self.b_shader.get_attrib("position"),
            3, GL_FLOAT, False, self.volume_stride, o.vtVBO
            )

        glEnableVertexAttribArray( self.b_shader.get_attrib("texcoord") )
        glVertexAttribPointer(
            self.b_shader.get_attrib("texcoord"),
            3, GL_FLOAT, False, self.volume_stride, o.vtVBO+12
            )

        glBindVertexArray( 0 )
        glDisableVertexAttribArray( self.b_shader.get_attrib("position") )
        glDisableVertexAttribArray( self.b_shader.get_attrib("texcoord") )

        o.elVBO=VBO(idx_out, target=GL_ELEMENT_ARRAY_BUFFER)
        o.elCount=len(idx_out.flatten())
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        sc = 1.0/la.norm(tl)
        c = 0.5*tl

        o.transform = np.array(( (0.0, 0.0, sc, -sc*c[2]), (0.0, sc, 0.0, -sc*c[1]),  (sc, 0.0, 0.0, -sc*c[0]), (0.0, 0.0, 0.0, 1.0)))

        return o

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
        glutPostRedisplay()

    def on_mouse_move(self, x, y, z=0):
        if self.moving:
            self.ex, self.ey = x, y
            self.ball.drag([x,y])
            glutPostRedisplay()

    def start(self):
        glutDisplayFunc(self.draw)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.key)
        glutMouseFunc(self.on_mouse_button)
        glutMouseWheelFunc(self.on_mouse_button)
        glutMotionFunc(self.on_mouse_move)

        glutMainLoop()

    def reshape(self, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        self.PMatrix = perspective(40.0, float(width)/height, 0.1, 10000.0)
        self.ball.place([width/2,height/2],height/2)
        self.init_back_texture()
        glutPostRedisplay()

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0,0.0,0.0,1.0)

        self.VMatrix = translation_matrix((0, 0, -self.dist)).dot(self.ball.matrix()).dot(scale_matrix(self.zoom))
        for obj in self.objs:
            self.render_volume_obj(obj)
        glutSwapBuffers()


    def init_back_texture(self):

        if self.fbo == None:
            self.fbo = glGenFramebuffers(1)
        print("fbo", self.fbo)

        glActiveTexture(GL_TEXTURE0 + 1)

        if self.bfTex != None:
            glDeleteTextures([self.bfTex])

        self.bfTex = glGenTextures(1)

        print("gen Tex 1")
        glBindTexture(GL_TEXTURE_2D, self.bfTex)

        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        print("bound", self.bfTex)

        print(self.width, self.height)
        w = int(self.width)
        h = int(self.height)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, None)
        print("made texture img")


        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        glFramebufferTexture2D(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, self.bfTex, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glBindTexture(GL_TEXTURE_2D, 0)

    def load_stack(self, stack_fn):
        data = open_tiff(stack_fn)

        print('data shape', data.shape)

        s = np.array(data, dtype=np.uint8, order='F')

        print(s.shape)

        w, h, d = s.shape
        print('shape', s.shape)

        stack_texture = glGenTextures(1)
        print(stack_texture)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, stack_texture)

        glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

       # glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
       # glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
       # glTexParameter(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, d, h, w, 0, GL_RED, GL_UNSIGNED_BYTE, s)
        print("made 3D texture")
        return stack_texture, s.shape

    def render_volume_obj(self, obj):

        vs = self.volume_shaders
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, obj.stack_texture)

        glClear(GL_COLOR_BUFFER_BIT)


        glEnable(GL_CULL_FACE)

        glCullFace(GL_BACK) #NB flipped

#        glValidateProgram(self.b_shader.program)
#        print("b_valid ", glGetProgramiv(self.b_shader.program, GL_VALIDATE_STATUS))
#        print(glGetProgramInfoLog(self.b_shader.program).decode())

        glUseProgram(self.b_shader.program)


        glBindVertexArray( obj.vao )
        print("copied", obj.elVBO.copied)
        obj.elVBO.bind()

        mv_matrix = np.dot(self.VMatrix, obj.transform)
        glUniformMatrix4fv(self.b_shader.get_uniform("mv_matrix"), 1, True, mv_matrix.astype('float32'))
        glUniformMatrix4fv(self.b_shader.get_uniform("p_matrix"), 1, True, self.PMatrix.astype('float32'))

        glDrawElements(
                GL_TRIANGLES, obj.elCount,
                GL_UNSIGNED_INT, obj.elVBO
            )

        obj.elVBO.unbind()
        glBindVertexArray( 0 )
        glUseProgram(0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glActiveTexture(GL_TEXTURE0+1)
        glBindTexture(GL_TEXTURE_2D, self.bfTex)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, obj.stack_texture)


        glUseProgram(vs.f_shader)

        glUniform1i(self.f_shader.get_uniform("texture3s"), 0)
        glUniform1i(self.f_shader.get_uniform("backfaceTex"), 1)


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )

        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)

        glBindVertexArray(obj.vao)
        obj.elVBO.bind()

        glUniformMatrix4fv(self.f_shader.get_uniform("mv_matrix"), 1, True, mv_matrix.astype('float32'))
        glUniformMatrix4fv(self.f_shader.get_uniform("p_matrix"), 1, True, self.PMatrix.astype('float32'))

        glDrawElements(
                GL_TRIANGLES, obj.elCount,
                GL_UNSIGNED_INT, obj.elVBO
            )

        glActiveTexture(GL_TEXTURE0+1)
        glBindTexture(GL_TEXTURE_2D, 0)

        glCullFace(GL_BACK)
        obj.elVBO.unbind()
        glBindVertexArray( 0 )
        glUseProgram(0)


    def key(self, k, x, y):
        if k=='+':
            self.zoom *= 1.1
        elif k=='-':
            self.zoom *= 0.9
        glutPostRedisplay()


def main():
    r = RenderWindow()
    if len(sys.argv)>=5:
        spacing = map(float, sys.argv[2:5])
    else:
        spacing = (1.0, 1.0, 1.0)
    r.initGL()
    r.objs.append(r.make_volume_obj(sys.argv[1], spacing))
    r.start()

if __name__ == '__main__':
    main()
