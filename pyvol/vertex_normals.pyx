

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

import cython
from cython.parallel import parallel, prange

@cython.boundscheck(False)
@cython.cdivision(True)
def calculate_vertex_normals(double[:,:] verts, unsigned int[:,:] tris):
    cdef int NV = verts.shape[0]
    cdef int NT = tris.shape[0]
    cdef int i, j
    cdef double[:,:] norms = np.zeros((NV,3), dtype=np.float64)
    cdef double[:] count = np.zeros((NV,), dtype=np.float64)
    cdef double[:] n = np.zeros((3,), dtype=np.float64)
    cdef double[:] nA = np.zeros((3,), dtype=np.float64)
    cdef double[:] d0 = np.zeros((3,), dtype=np.float64)
    cdef double[:] d1 = np.zeros((3,), dtype=np.float64)
    cdef double l
    cdef unsigned int i0, i1, i2

    with nogil:
        for i in range(NT):
            i0 = tris[i,0]
            i1 = tris[i,1]
            i2 = tris[i,2]
            for j in range(3):
                d0[j] = verts[i1,j] - verts[i0,j]
            for j in range(3):
                d1[j] = verts[i2,j] - verts[i0,j]
            nA[0] = d0[1]*d1[2] - d0[2]*d1[1]
            nA[1] = d0[2]*d1[0] - d0[0]*d1[2]
            nA[2] = d0[0]*d1[1] - d0[2]*d1[0]
            for j in range(3):
                count[tris[i,j]] += 1.0
                for k in range(3):
                    norms[tris[i,j],k] += nA[k]
    
        for i in prange(NV):
            l = sqrt(norms[i,0]*norms[i,0] + norms[i,1]*norms[i,1] + norms[i,2]*norms[i,2])
            for j in range(3):
                norms[i,j] = norms[i,j]/l
    return norms
        
            
    
