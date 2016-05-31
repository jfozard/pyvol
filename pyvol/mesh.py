"""Module with mesh classes."""

import numpy as np
import numpy.linalg as la

from ply_parser import parse_ply2

class ProjectionMesh(object):
    def __init__(self):
        pass

    def load_ply2(self, fn):
        descr, data = parse_ply2(fn)
        self.descr = descr
        self.data = data
        NV = len(data['vertex'][0])
        NF = len(data['face'][0])
        print('NF', NF)
        verts = []
        vert_norms = []
        print(data['vertex'][1])
        x_idx = data['vertex'][1].index('x')
        y_idx = data['vertex'][1].index('y')
        z_idx = data['vertex'][1].index('z')
        nx_idx = data['vertex'][1].index('nx')
        ny_idx = data['vertex'][1].index('ny')
        nz_idx = data['vertex'][1].index('nz')
        for v in data['vertex'][0]:
            verts.append((v[x_idx], v[y_idx], v[z_idx]))
            vert_norms.append(np.array((v[nx_idx], v[ny_idx], v[nz_idx])))
        print(('done_vertex'))
        v_array=np.array(verts,dtype='float32')

        self.bbox=(np.min(v_array,0),  np.max(v_array,0) )
        self.verts = [np.array(v) for v in v_array]
        self.vert_norms = vert_norms

        self.tris = []
        for f in data['face'][0]:
            vv = f[0]
            tris = []
            for i in range(len(vv)-2):
                tris.append((vv[0], vv[i+1], vv[i+2]))
            self.tris.extend(tris)
        print('done_face')

    def get_zoom(self):
        zoom=1.0/la.norm(self.bbox[1]-self.bbox[0])
        return zoom

    def generate_arrays_projection(self):
        print(('start_arrays'))
        npr.seed(1)
        tris = []
        v_out=np.array(self.verts,dtype=np.float32)
        idx_out=np.array(self.tris,dtype=np.uint32)
        n_out=np.array(self.vert_norms,dtype=np.float32)
#        col_out = np.tile(nsignal,(3,1)).T
        col_out = np.ones(v_out.shape, dtype=np.float32)
        return v_out, n_out, col_out, idx_out
