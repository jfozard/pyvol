import logging

import numpy as np
import scipy.linalg as la

from algo import calculate_vertex_normals
from mesh import Mesh

class GLMesh(Mesh):
    def __init__(self):
        Mesh.__init__(self)

    def load_ply(self, filename):
        Mesh.load_ply(self, filename)

        v_array=np.array(self.verts, dtype=np.float32)
        bbox=(np.min(v_array,0),  np.max(v_array,0) )

        self.v_array = v_array.astype(np.float32)
        self.bbox = bbox
        self.zoom=1.0/la.norm(bbox[1])

        self.tri_array = np.array(self.tris, dtype=np.uint32)
        self.n_array = self.vert_props['normal'].astype(np.float32)

        logging.debug( 'done matrix {}'.format(self.zoom) )
        return self.zoom


    def generate_arrays(self):

        tris = []
        v_out = self.v_array
        idx_out = self.tri_array
        n_out = self.n_array

        if 'color' in self.vert_props:
            col_out = (self.vert_props['color']/255.0).astype(np.float32)
            logging.debug( col_out )
        elif 'signal' in self.vert_props:
            signal = self.vert_props['signal']

            nsignal = 0.1+ 0.9*(signal / float(np.max(signal)))
            nsignal = nsignal.astype(np.float32)
            col_out = np.tile(nsignal,(3,1)).T
        else:
            col_out = (0.5*np.ones(v_out.shape)).astype(np.float32)
        return v_out, n_out, col_out, idx_out
