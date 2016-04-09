
import sys
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

from meshutils.mesh.mesh import Mesh, calculate_vertex_normals

from scipy.sparse import dok_matrix, csr_matrix

import scipy.ndimage as nd

from transformations import Arcball

from scipy.sparse import dok_matrix, csr_matrix


from PIL import Image

from scipy.interpolate import RectBivariateSpline
from itertools import chain

import matplotlib.pyplot as plt

from max_project import max_project

from mesh_reproject import mesh_reproject

class Obj():
    pass

def open_tiff(fn):
    im = Image.open(fn)
    frames = []
    i = 0
    try:
        while True:
            im.seek(i)
            # i2 = np.sum(np.asarray(im), axis=2)
            i2 = np.asarray(im)
            frames.append(i2)
            i += 1
    except EOFError:
        pass

    im = np.dstack(frames)
    del frames
    return im

def translate(x,y,z):
    a = np.eye(4)
    a[0,3]=x
    a[1,3]=y
    a[2,3]=z
    return a

def scale(s):
    a = np.eye((4))
    a[0,0]=s
    a[1,1]=s
    a[2,2]=s
    a[3,3]=1.0
    return a

def perspective(fovy, aspect, zNear, zFar):
    f = 1.0/math.tan(fovy/2.0/180*math.pi)
    return np.array(((f/aspect, 0, 0, 0), (0,f,0,0), (0,0,(zFar+zNear)/(zNear-zFar), 2*zFar*zNear/(zNear-zFar)), (0, 0, -1, 0)))

def link_shader_program(vertex_shader, fragment_shader):
    """Create a shader program with from compiled shaders."""
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    # check linking error
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not(result):
        raise RuntimeError(glGetProgramInfoLog(program))
    return program


class ProjectionMesh(Mesh):
    def __init__(self):
        Mesh.__init__(self)

    def make_connectivity_matrix(self):
        n = len(self.verts)
        connections = {}
        for t in self.tris:
            connections[(t[0], t[1])] = 1
            connections[(t[1], t[0])] = 1
            connections[(t[1], t[2])] = 1
            connections[(t[2], t[1])] = 1
            connections[(t[2], t[0])] = 1
            connections[(t[0], t[2])] = 1
        A = dok_matrix((n,n))
        A.update(connections)
        A = A.tocsr()
        D = A.sum(axis=1)
        self.connectivity = A
        self.degree = D
#        print 'bad_verts', np.where(D==0)


    def load_ply2(self, fn):
  
        Mesh.load_ply(self, fn)

        v_array=np.array(self.verts)
        self.bbox=(np.min(v_array,0),  np.max(v_array,0) )
 


    def set_geom(self, verts, tris=None):
        self.verts = np.asarray(verts, dtype=float)
        if tris is not None:
            self.tris = np.asarray(tris, dtype=int)
            self.make_connectivity_matrix()

        # Calculate mesh normals
        self.bbox=(np.min(self.verts,0),  np.max(self.verts,0) )
        print 'bbox', self.bbox
        self.vert_props = { 'normal': calculate_vertex_normals(self.verts, self.tris) }


    def generate_arrays_projection(self, signal=False):
        v_out=np.array(self.verts,dtype=np.float32) 
        idx_out=np.array(self.tris,dtype=np.uint32)
        n_out=np.array(self.vert_props['normal'],dtype=np.float32)
        col_out = np.ones(v_out.shape, dtype=np.float32)
        return v_out, n_out, col_out, idx_out

    def reproject(self, stack, spacing):
        print 'shape', stack.shape, stack.strides
        print 'Stack range', np.max(stack), np.min(stack)
        print 'bbox', np.min(self.verts, axis=0), np.max(self.verts, axis=0)
        print 'spacing', spacing, spacing.shape

        mesh_reproject(stack, mesh, spacing, -50, 10, 101)
        
        """
        vert_signal = np.zeros((len(self.verts),), dtype=float)
        vert_norms = self.vert_props['normal']

        print 'verts shape', self.verts.shape
        print 'vn shape', vert_norms.shape

        for i in range(self.verts.shape[0]):
            v = self.verts[i]
            n = vert_norms[i]
            p_start = v + 10*n
            p_end = v - 50*n
            p_start = p_start/spacing
            p_end = p_end/spacing
            #print v, n
            #print p_start, p_end
            tt = np.linspace(0, 1, 101)
            res = nd.map_coordinates(stack,
                                     ([(p_start[0]*(1-t) + p_end[0]*t) for t in tt],
                                      [(p_start[1]*(1-t) + p_end[1]*t) for t in tt],
                                      [(p_start[2]*(1-t) + p_end[2]*t) for t in tt]), order=1 )
            j = np.argmax(res)
            t = tt[j]
            vert_signal[i] = res[j]
            self.verts[i] = (v+10*n)*(1-t) + (v-50*n)*t
            if i%1000==0:
                print j, np.max(res)
        
        self.vert_props['signal'] = vert_signal
        """

    def project(self, stack, spacing, d0=0, d1=-0.1, samples=10, op=np.mean):
        print 'shape', stack.shape
        print 'Stack range', np.max(stack), np.min(stack)
        print 'bbox', np.min(self.verts, axis=0), np.max(self.verts, axis=0)
        print 'spacing', spacing

        verts = self.verts
        vert_signal = np.zeros((len(self.verts),), dtype=float)
        vert_norms = self.vert_props['normal']

        mesh_project(stack, mesh, spacing, d0, d1, samples)
        """
        tt = np.linspace(0, 1, samples)
        for i in range(self.verts.shape[0]):
            v = verts[i]
            n = vert_norms[i]
            p_start = v + d0*n
            p_end = v + d1*n
            p_start = p_start/spacing
            p_end = p_end/spacing
            #print p_start, p_end

            vert_signal[i] = op(nd.map_coordinates(stack,
                                     ([(p_start[0]*(1-t) + p_end[0]*t) for t in tt],
                                      [(p_start[1]*(1-t) + p_end[1]*t) for t in tt],
                                      [(p_start[2]*(1-t) + p_end[2]*t) for t in tt]), order=1 ))
            if i%1000==0:
                print i, vert_signal[i]
        
        self.vert_props['signal'] = vert_signal
        """


    def split_long_edges(self, lc):

        def sort_pair(a, b):
            if a<=b:
                return a,b
            else:
                return b,a

        verts = self.verts
        tris = self.tris
        verts = [ v for v in verts ]
        new_tris = []
        
        # Recalculate normals explicitly, reproject to get signal


        new_point_map = {}

        # Loop over all triangles; find edge lengths
        for t in tris:
            # Measure triangle edge lengths
            i0, i1, i2 = t
            split = []
            x0 = np.asarray(verts[i0])
            x1 = np.asarray(verts[i1])
            x2 = np.asarray(verts[i2])
            d0 = la.norm(x2-x1)
            if d0>lc:
                pp = sort_pair(i1, i2)
                try:
                    s0 = new_point_map[pp]
                except KeyError:
                    m0 = 0.5*(x1+x2)
                    s0 = len(verts)
                    new_point_map[pp] = s0
                    verts.append(m0)

                split.append((0, s0))


            d1 = la.norm(x0-x2)
            if d1>lc:
                pp = sort_pair(i0, i2)
                try:
                    s1 = new_point_map[pp]
                except KeyError:
                    m1 = 0.5*(x0+x2)
                    s1 = len(verts)
                    new_point_map[pp] = s1
                    verts.append(m1)
                split.append((1, s1))



            d2 = la.norm(x1-x0)
            if d2>lc:
                pp = sort_pair(i0, i1)
                try:
                    s2 = new_point_map[pp]
                except KeyError:
                    m2 = 0.5*(x0+x1)
                    s2 = len(verts)
                    new_point_map[pp] = s2
                    verts.append(m2)
                split.append((2, s2))

            N = len(split)
            if N==0:
                new_tris.append(t)
            elif N==1:
                s = split[0][0]
                idx_s = t[s]
                idx_sp = t[(s+1)%3]
                idx_sm = t[(s+2)%3]
                idx_o = split[0][1]
                new_tris.append((idx_s, idx_sp, idx_o))
                new_tris.append((idx_s, idx_o, idx_sm))
            elif N==2:
                d = dict(split)
                if 0 not in d:
                    s1 = d[1]
                    s2 = d[2]
                    new_tris.append((t[0], s2, s1))
                    new_tris.append((t[2], s1, s2))
                    new_tris.append((t[1], t[2], s2))
                elif 1 not in d:
                    s0 = d[0]
                    s2 = d[2]
                    new_tris.append((t[1], s0, s2))
                    new_tris.append((t[0], s2, s0))
                    new_tris.append((t[2], t[0], s0))
                else:
                    s0 = d[0]
                    s1 = d[1]
                    new_tris.append((t[2], s1, s0))
                    new_tris.append((t[1], s0, s1))
                    new_tris.append((t[0], t[1], s1))
            elif N==3:
                split_tri = [_[1] for _ in split]
                new_tris.append((t[0], split_tri[2], split_tri[1]))
                new_tris.append((t[1], split_tri[0], split_tri[2]))
                new_tris.append((t[2], split_tri[1], split_tri[0]))
                new_tris.append(split_tri)

        
        self.set_geom(verts, new_tris)


    def clip_triangles(self, zheight):
        # Remove triangles for which all three vertices are above zheight
        verts = self.verts
        vert_mask = [i for i, x in enumerate(verts) if x[2]<zheight]
        vert_set = set(vert_mask)
        new_tris = []
        for t in mesh.tris:
            if not all(i not in vert_set for i in t):
                new_tris.append(t)
        retained_verts = set(chain.from_iterable(new_tris))
        retained_verts = sorted(retained_verts)
        vert_map = dict((j,i) for i, j in enumerate(retained_verts))

        tris = np.array([map(vert_map.get, t) for t in new_tris], dtype=int)
        verts = np.array([verts[i] for i in retained_verts])
        
        self.set_geom(verts, tris)

    def mean_tri_edge_length(self):
        verts = self.verts
        tris = self.tris
        edge_tot = 0.0
        sqrt = math.sqrt
        for t in self.tris:
            for j in range(3):
                d = verts[t[(j+1)%3],:] - verts[t[j],:]
                edge_tot += sqrt(np.dot(d,d))
        return edge_tot / (3*self.tris.shape[0])
        

    def smooth_surface(self, delta=0.05, iterations=10):
        A = self.connectivity
        D = self.degree
        
        for i in range(iterations):
            self.verts = np.asarray((1-delta)*self.verts + delta*A.dot(self.verts)/(1e-6 + D))
        self.set_geom(self.verts)

    @classmethod
    def from_data(cls, verts, tris):
        m = ProjectionMesh()
        m.set_geom(verts, tris)
        return m
                                     

def make_surface(ps, ma, spacing, mesh2, dm=-20, dp=22):
    # write the projection surface to an off file

    verts = []
    
    h = RectBivariateSpline(range(ps.shape[0]), range(ps.shape[1]), ps)
        
    for v in mesh2.verts:
        x = v[0]/spacing[0]
        y = v[1]/spacing[1]

        Z = h(x,y)[0][0]*spacing[2]

        verts.append((v[0], v[1], Z))

    tris = []
    for t in mesh2.tris:
        tris.append(list(t))


    m = ProjectionMesh.from_data(verts, tris)

    return m



def make_square_triangulation(ps, spacing, m=None, n=None):

    if m==None:
        m = ps.shape[0]
    if n==None:
        n = ps.shape[1]

    NV = m*n
    
    verts = []
    tris = []

    for i in np.linspace(0, ps.shape[0]-1, m):
        for j in np.linspace(0, ps.shape[1]-1, n):
            verts.append((i*spacing[0], j*spacing[1], 0.0))

    print len(verts), NV

    for i in range(m-1):
        for j in range(n-1):
            tris.append((i*n+j, i*n+(j+1), (i+1)*n+j))
            tris.append(((i+1)*n+j, i*n+(j+1), (i+1)*n+(j+1)))

    return ProjectionMesh.from_data(verts, tris)


class RenderWindow(object):
    def __init__(self):
        glutInit([])
        glutInitContextVersion(3, 2)
        glutInitWindowSize(800, 600)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH)
        self.window = glutCreateWindow("Cell surface")
        self.renderer = Renderer()

class Renderer(object):
    def __init__(self):
        self.width = 800
        self.height = 600
        self.PMatrix = np.eye(4)
        self.VMatrix = np.eye(4)
        self.volume_objs = []
        self.project_objs = []
        self.solid_objs = []
        self.moving = False
        self.bfTex = None
        self.fbo = None
        self.render_volume = True

    def initGL(self):
        self.ball = Arcball()
        self.zoom = 0.5
        self.dist = 2.0

        self.make_volume_shaders()
        self.make_project_shaders()
        
        self.reshape(self.width, self.height)

    def make_volume_shaders(self):

        vs = Obj()
        vertex = compileShader(
            """
	    attribute vec3 position;
	    attribute vec3 texcoord;

            varying vec3 v_texcoord;
            varying vec4 v_pos;

            uniform mat4 mv_matrix;
            uniform mat4 p_matrix;
	    void main() {
                vec4 eye =  mv_matrix * vec4(position,1.0);
		v_pos = p_matrix * eye;
                gl_Position = v_pos;
                v_texcoord = texcoord;
	    }""",
            GL_VERTEX_SHADER)
        
        front_fragment = compileShader(
            """
            varying vec3 v_texcoord;
            varying vec4 v_pos;

            uniform sampler2D backfaceTex;            
            uniform sampler3D texture_3d;
            const float falloff = 0.995;

	    void main() {
                vec2 texc = (v_pos.xy/v_pos.w +1.0)/2.0; //((/gl_FragCoord.w) + 1) / 2;
                vec3 startPos = v_texcoord;
                vec3 endPos = texture2D(backfaceTex, texc).rgb;
                vec3 ray = endPos - startPos;
                float rayLength = length(ray);
                vec3 step = normalize(ray)*(2.0/600.0);
                vec4 colAcc = vec4(0,0,0,0);
                float sample;
                vec3 samplePos = vec3(0,0,0); 
                for (int i=0; i<1000; i++)
                {
                    sample = texture3D(texture_3d, endPos - samplePos).x;
                    colAcc.rgb = mix(colAcc.rgb, vec3(1.0, 0.0, 0.0), sample*0.1);
                    colAcc.a = mix(colAcc.a, 1.0, sample*0.1);
                    colAcc *= falloff;

                    if ((length(samplePos) >= rayLength))
                        break;
                    //if(colAcc.a>0.99) {
                    //    colAcc.a = 1.0;
                    //    colAcc.rgb = vec3(0,1,0);
                    //    break;
                    //}
                    samplePos += step;
                }
                gl_FragColor = colAcc;

	    }""",
            GL_FRAGMENT_SHADER)


        back_fragment = compileShader(
            """
            varying vec3 v_texcoord;            

	    void main() {
                    gl_FragColor = vec4(v_texcoord,1.0);
	    }""",
            GL_FRAGMENT_SHADER)
            
        vs.b_shader = link_shader_program(vertex, back_fragment)

        vs.b_position_location = glGetAttribLocation( 
            vs.b_shader, 'position' 
            )
        vs.b_texcoord_location = glGetAttribLocation( 
            vs.b_shader, 'texcoord' 
            )

        vs.b_mv_location = glGetUniformLocation(
            vs.b_shader, 'mv_matrix'
            )
        vs.b_p_location = glGetUniformLocation(
            vs.b_shader, 'p_matrix'
            )

        vs.vStride = 6*4

        vs.f_shader = link_shader_program(vertex, front_fragment)

        vs.f_position_location = glGetAttribLocation( 
            vs.f_shader, 'position' 
            )
        vs.f_texcoord_location = glGetAttribLocation( 
            vs.f_shader, 'texcoord' 
            )

        vs.f_mv_location = glGetUniformLocation(
            vs.f_shader, 'mv_matrix'
            )
        vs.f_p_location = glGetUniformLocation(
            vs.f_shader, 'p_matrix'
            )

        vs.f_bfTex_location = glGetUniformLocation(
            vs.f_shader, 'backfaceTex'
            )
        vs.f_t3d_location = glGetUniformLocation(
            vs.f_shader, 'texture3d'
            )
        self.volume_shaders = vs

    def make_project_shaders(self):

        ps = Obj()

        project_vertex = compileShader(
            """
            attribute vec3 position;
	    attribute vec3 normal;
            attribute vec3 color;

            varying vec3 v_texcoord;
            varying vec3 v_texnormal;
            
            varying vec3 v_normal;
            varying vec3 v_color;

            uniform mat4 tex_matrix;
            uniform mat4 mv_matrix;
            uniform mat4 p_matrix;

            uniform float depth_start;

            uniform bool move_surface;

	    void main() {
                vec3 new_pos;
                if(move_surface) {
                    new_pos = position + depth_start*normal;
                } else {
                    new_pos = position;
                }
                vec4 eye =  mv_matrix * vec4(new_pos, 1.0);
                v_color = color;
                v_normal = (mv_matrix * vec4(normal, 0.0)).xyz;

                v_texcoord = (tex_matrix *vec4(position, 1.0)).xyz;
                v_texnormal = (tex_matrix *vec4(normalize(normal), 0.0)).xyz;

                gl_Position = p_matrix * eye;

	    }""",
            GL_VERTEX_SHADER)
        
        project_fragment = compileShader(
            """
            varying vec3 v_texcoord;
            varying vec3 v_texnormal;
            
            varying vec3 v_color;
            varying vec3 v_normal;

            uniform mat4 tex_matrix;

            uniform sampler3D texture_3d;

            uniform float depth_start;
            uniform float depth_end;
            
            const vec3 light_direction =  vec3(0., 0., -1.);       
            const vec4 light_diffuse = vec4(0.7, 0.7, 0.7, 0.0);
            const vec4 light_ambient = vec4(0.3, 0.3, 0.3, 1.0);   

            uniform float sample_gain;
            uniform float alpha_project;


	    void main() {
                vec3 tn = v_texnormal;
                vec3 startPos = v_texcoord + depth_start*tn;
                vec3 step = (depth_end - depth_start)*tn/19.0;
                vec4 colAcc = vec4(0,0,0,0);
                vec3 currentPos = startPos;
            
                float total_sample = 0.0;
                // Sample stack (3D texture) along ray
                for (int i=0; i<10; i++)
                {
                    total_sample += texture3D(texture_3d, currentPos.xyz).x;
                    currentPos += step;
                }
                // Average and scale samples
                float mean_sample = clamp(0.1*sample_gain*total_sample, 0.0, 1.0);

                vec4 projected_color = vec4(startPos.y, mean_sample, 0.0, 1.0);
                // Find surface color
                vec3 normal = normalize(v_normal);
                vec4 diffuse_factor = max(-dot(normal, light_direction), 0.0) * light_diffuse;
                vec4 diffuse_color = (diffuse_factor + light_ambient)*vec4(v_color, 1.0);
                // Combine surface and projected color
                gl_FragColor = mix(projected_color, diffuse_color, alpha_project);
                
	    }""",
            GL_FRAGMENT_SHADER)

        ps.shader = link_shader_program(project_vertex, project_fragment)

        ps.position_location = glGetAttribLocation( 
            ps.shader, 'position' 
            )

        ps.normal_location = glGetAttribLocation( 
            ps.shader, 'normal' 
            )

        ps.color_location = glGetAttribLocation( 
            ps.shader, 'color' 
            )


        ps.mv_location = glGetUniformLocation(
            ps.shader, 'mv_matrix'
            )
        ps.p_location = glGetUniformLocation(
            ps.shader, 'p_matrix'
            )

        ps.tex_location = glGetUniformLocation(
            ps.shader, 'tex_matrix'
            )

        ps.t3d_location = glGetUniformLocation(
            ps.shader, 'texture3d'
            )

        ps.depth_start_location = glGetUniformLocation(
            ps.shader, 'depth_start'
            )

        ps.depth_end_location = glGetUniformLocation(
            ps.shader, 'depth_end'
            )

        ps.sample_gain_location = glGetUniformLocation(
            ps.shader, 'sample_gain'
            )

        ps.alpha_project_location = glGetUniformLocation(
            ps.shader, 'alpha_project'
            )

        ps.move_surface_location = glGetUniformLocation( 
            ps.shader, 'move_surface' 
            )

        ps.vStride = 9*4
        self.project_shader = ps


    def make_stack_obj(self, data, spacing):
        so = Obj()
        so.stack_texture, so.data, so.shape = self.load_stack(data)
        so.spacing = np.array(spacing)
        return so

    def make_volume_obj(self, so):
        o = Obj()        

        o.so = so

        o.vao = glGenVertexArrays(1)
        glBindVertexArray(o.vao)
        vs = self.volume_shaders

        tl = np.array((so.shape[2]*so.spacing[2],
                       so.shape[1]*so.spacing[1],
                       so.shape[0]*so.spacing[0]))

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

        glEnableVertexAttribArray( vs.b_position_location )
        glVertexAttribPointer( 
            vs.b_position_location, 
            3, GL_FLOAT, False, vs.vStride, o.vtVBO 
            )

        glEnableVertexAttribArray( vs.b_texcoord_location )
        glVertexAttribPointer( 
            vs.b_texcoord_location, 
            3, GL_FLOAT, False, vs.vStride, o.vtVBO+12
            )

        glBindVertexArray( 0 )
        glDisableVertexAttribArray( vs.b_position_location )
        glDisableVertexAttribArray( vs.b_texcoord_location )

        o.elVBO=VBO(idx_out, target=GL_ELEMENT_ARRAY_BUFFER)
        o.elCount=len(idx_out.flatten())
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        sc = 1.0/la.norm(tl)
        c = 0.5*tl

        o.transform = np.array(( (0.0, 0.0, sc, -sc*c[2]), (0.0, sc, 0.0, -sc*c[1]),  (sc, 0.0, 0.0, -sc*c[0]), (0.0, 0.0, 0.0, 1.0)))

        return o

    def make_project_obj(self, mesh, so):

        o = Obj()
        o.so = so

        o.mesh = mesh

        v_out, n_out, col_out, idx_out  = o.mesh.generate_arrays_projection()

        vb=np.concatenate((v_out,n_out,col_out),axis=1)

        vao = glGenVertexArrays(1)

        glBindVertexArray(vao)
        print "made VAO"
        
        o.mesh_vtVBO=VBO(vb)

        print 'made VBO'
        o.mesh_vtVBO.bind()

        ps = self.project_shader

        glEnableVertexAttribArray( ps.position_location )
        glVertexAttribPointer( 
            ps.position_location, 
            3, GL_FLOAT, False, ps.vStride, o.mesh_vtVBO 
            )

        glEnableVertexAttribArray( ps.normal_location )
        glVertexAttribPointer( 
            ps.normal_location, 
            3, GL_FLOAT, False, ps.vStride, o.mesh_vtVBO+12
            )

        glEnableVertexAttribArray( ps.color_location )
        glVertexAttribPointer( 
            ps.color_location, 
            3, GL_FLOAT, False, ps.vStride, o.mesh_vtVBO+24
            )


        glBindVertexArray( 0 )
        glDisableVertexAttribArray( ps.position_location )
        glDisableVertexAttribArray( ps.normal_location )
        glDisableVertexAttribArray( ps.color_location )
        glBindBuffer(GL_ARRAY_BUFFER, 0)


        o.mesh_elVBO=VBO(idx_out, target=GL_ELEMENT_ARRAY_BUFFER)
        o.mesh_elCount=len(idx_out.flatten())
        o.mesh_vao = vao

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        print o.mesh_elCount

        print 'made obj', vao
        
        bbox = o.mesh.bbox

        print 'mesh bbox', o.mesh.bbox
        c = 0.5*(bbox[1] + bbox[0])
        sc = 1.0/la.norm(bbox[1] - bbox[0])
        
        print 'mesh centre', c, sc
        
        #tl = np.array((512*1.08, 512.0*1.08, 955*0.25))
        tl = np.array(o.so.spacing)*np.array(o.so.shape)
        print o.so.spacing, o.so.shape, tl,  np.array((512*1.08, 512.0*1.08, 955*0.25))

        c = 0.5*tl
        sc = 1.0/la.norm(tl)
        
        o.transform = np.array(((sc, 0.0, 0.0, -sc*c[0]), (0.0, sc, 0.0, -sc*c[1]), (0.0, 0.0, sc, -sc*c[2]), (0.0, 0.0, 0.0, 1.0)))
#        o.tex_transform = np.array(((1.0/tl[0], 0.0, 0.0, 0.0), (0.0, 1.0/tl[1], 0.0, 0.0), (0.0, 0.0, 1.0/tl[2], 0.0), (0.0, 0.0, 0.0, 1.0)))
        o.tex_transform = np.array(( (0.0, 0.0, 1.0/tl[2], 0.0),(0.0, 1.0/tl[0], 0.0, 0.0), (1.0/tl[1], 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))


        return o



    def update_project_obj(self, o):

        v_out, n_out, col_out, idx_out  = o.mesh.generate_arrays_projection(signal=True)
        vb = np.concatenate((v_out,n_out,col_out),axis=1)
        
        o.mesh_vtVBO.bind()
        o.mesh_vtVBO.set_array(vb)
        o.mesh_vtVBO.copy_data()
        o.mesh_vtVBO.unbind()

        o.mesh_elVBO.set_array(idx_out)
        o.mesh_elCount = len(idx_out.flatten())



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

        self.VMatrix = translate(0, 0, -self.dist).dot(self.ball.matrix()).dot(scale(self.zoom))
        
        if not self.render_volume:
            for obj in self.project_objs:
                self.render_project_obj(obj)
        else:
            for obj in self.volume_objs:
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

    def load_stack(self, data):

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
        return stack_texture, data, s.shape

    def render_volume_obj(self, obj):

        vs = self.volume_shaders
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, obj.so.stack_texture)

        glClear(GL_COLOR_BUFFER_BIT)

        glEnable(GL_CULL_FACE)

        
        glCullFace(GL_BACK) #NB flipped

        glUseProgram(vs.b_shader)

        glBindVertexArray( obj.vao )
        print("copied", obj.elVBO.copied)
        obj.elVBO.bind()

        mv_matrix = np.dot(self.VMatrix, obj.transform)
        glUniformMatrix4fv(vs.b_mv_location, 1, True, mv_matrix.astype('float32'))
        glUniformMatrix4fv(vs.b_p_location, 1, True, self.PMatrix.astype('float32'))

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
        glBindTexture(GL_TEXTURE_3D, obj.so.stack_texture)


        glUseProgram(vs.f_shader)

        glUniform1i(vs.f_t3d_location, 0)
        glUniform1i(vs.f_bfTex_location, 1)


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )

        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT) 

        glBindVertexArray(obj.vao)
        obj.elVBO.bind()

        glUniformMatrix4fv(vs.f_mv_location, 1, True, mv_matrix.astype('float32'))
        glUniformMatrix4fv(vs.f_p_location, 1, True, self.PMatrix.astype('float32'))

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


    def render_project_obj(self, obj):


#        glActiveTexture(GL_TEXTURE0)
#        glEnable(GL_TEXTURE_3D)
#        glBindTexture(GL_TEXTURE_3D, obj.stack.stack_texture)

        glDepthMask(True)

#        print self.bfTex, self.stack_texture

        ps = self.project_shader

  
        glUseProgram(ps.shader)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, obj.so.stack_texture)

        glUniform1i(ps.t3d_location, 0)

        glUniform1f(ps.depth_start_location, 0.0)
        glUniform1f(ps.depth_end_location, 1.0)
        glUniform1f(ps.sample_gain_location, 10.0)
        glUniform1f(ps.alpha_project_location, 0.5)
        glUniform1i(ps.move_surface_location, 1)


        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

#        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )

        
        glBindVertexArray( obj.mesh_vao )

        obj.mesh_elVBO.bind()


        mv_matrix = np.dot(self.VMatrix, obj.transform)

        glUniformMatrix4fv(ps.mv_location, 1, True, mv_matrix.astype('float32'))
        glUniformMatrix4fv(ps.p_location, 1, True, self.PMatrix.astype('float32'))

        glUniformMatrix4fv(ps.tex_location, 1, True, obj.tex_transform.astype('float32'))

#        print 'start_draw'
        glDrawElements(
                GL_TRIANGLES, obj.mesh_elCount,
                GL_UNSIGNED_INT, obj.mesh_elVBO
            )
#        print 'done_draw'


        obj.mesh_elVBO.unbind()
        glBindVertexArray( 0 )
        glUseProgram(0)




    def key(self, k, x, y):
        if k=='+':
            self.zoom *= 1.1
        elif k=='-':
            self.zoom *= 0.9
        elif k=='p':
            o = self.project_objs[0]
            m = o.mesh
            m.reproject(o.so.data, o.so.spacing)
            self.update_project_obj(o)
        elif k=='s':
            o = self.project_objs[0]
            m = o.mesh
            m.smooth_surface()
            self.update_project_obj(o)
        elif k=='r':
            o = self.project_objs[0]
            m = o.mesh
            print 'before - NV', m.verts.shape[0], 'NT', m.tris.shape[0]
            m.split_long_edges(0.0)
            print 'after  - NV', m.verts.shape[0], 'NT', m.tris.shape[0]
            self.update_project_obj(o)
        elif k=='t':
            o = self.project_objs[0]
            m = o.mesh
            print 'before - NV', m.verts.shape[0], 'NT', m.tris.shape[0]
            h = m.mean_tri_edge_length()
            m.split_long_edges(1.5*h)
            print 'after  - NV', m.verts.shape[0], 'NT', m.tris.shape[0]
            self.update_project_obj(o)

        elif k=='c':
            o = self.project_objs[0]
            m = o.mesh
            m.clip_triangles(o.so.spacing[2]*o.so.shape[2]*0.95)
            print 'NV', m.verts.shape[0]
            self.update_project_obj(o)

        elif k=='w':
            o = self.project_objs[0]
            m = o.mesh
            m.project(o.so.data, o.so.spacing)
            m.save_ply(sys.argv[2])
        elif k==' ':
            self.render_volume = not self.render_volume

        glutPostRedisplay()



#from image_io.import_tiff import load_tiff_stack, stop_javabridge


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    

def run_tiff(ma, spacing):

    spacing = [spacing[0], spacing[1], spacing[2]]

    
    pp = np.max(ma, axis=2)
    pp = nd.gaussian_filter(pp, 1.0)
    mask2d = pp>0.1*np.mean(pp)
    ci, cj = nd.center_of_mass(mask2d)

    area = spacing[0]*spacing[1]*np.sum(mask2d)


    print "area " +str(area)+"\n"

    cell_size = math.sqrt(area/400.0)
    
    cell_px = cell_size/spacing[0]

    print "estimated cell_size", cell_size
    print "estimated cell_px", cell_px


    bl1_scale = [0.5*cell_px, 0.5*cell_px, 0.3*cell_size/spacing[2]]

    bl1 = nd.gaussian_filter(ma, bl1_scale)

    bl1 = np.ascontiguousarray(np.transpose(bl1, (2, 0, 1)))

    ps = np.zeros((ma.shape[0], ma.shape[1]))

    m = np.mean(bl1, axis=0)
    b = 1.0
    c = 4.7
    t = np.maximum(b*m,c)
    r = 0.5

    max_project(bl1, t, ps)

#    plt.imshow(ps)
#    plt.show()
 
    phi = np.tanh((ma.shape[0] - 1 - ps) / (0.1*(ma.shape[1]-1)))
    bl_phi = nd.gaussian_filter(phi, r*cell_px)
    sps = np.minimum((phi/(bl_phi+1e-6))*nd.gaussian_filter(ps, r*cell_px)+(1.0-phi)*ps, ma.shape[0]-1) 

    depth = 20
    T = make_square_triangulation(ps, spacing, int(0.25*ma.shape[0]), int(0.25*ma.shape[1]))
    m = make_surface(ps, ma, spacing, T)

    return m

if __name__=='__main__':

    ma = open_tiff(sys.argv[1])

    rw = RenderWindow()
    r = rw.renderer
    if len(sys.argv)>=5:
        spacing = map(float, sys.argv[2:5])
    else:
        spacing = (1.0, 1.0, 0.25)
    r.initGL()


    so = r.make_stack_obj(ma, spacing)
    mesh = run_tiff(ma, spacing)
    r.volume_objs.append(r.make_volume_obj(so))
    r.project_objs.append(r.make_project_obj(mesh, so))
#    stop_javabridge()
    r.start()


        


if __name__ == '__main__': 
    main()



