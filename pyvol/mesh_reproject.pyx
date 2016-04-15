cimport cython
cimport numpy
from cython.parallel import parallel, prange


cdef extern void _mesh_reproject(unsigned char* A, int I, int J, int K, double* spacing, double* verts, double* norms,  int NV, unsigned int* tris, int NT, double level, double t, double s)

cpdef int mesh_reproject(A, mesh, spacing, level, t, s):
    cdef unsigned char[:,:,:] stack = A
    cdef double[:,:] verts = mesh.verts
    cdef double[:,:] norms = mesh.vert_props['normal']
    cdef double[:] sp = spacing
    cdef unsigned int[:,:] tris = mesh.tris

    print level, t, s
    _mesh_reproject(&stack[0,0,0], stack.shape[0], stack.shape[1], stack.shape[2], &sp[0], &verts[0,0], &norms[0,0], verts.shape[0], &tris[0,0], tris.shape[0], level, t, s)
    return 0


cdef extern void _mesh_project_mean(unsigned char* A, int I, int J, int K, double* spacing, double* verts, double* norms, double* signal, int NV, double d0, double d1, int n)

cpdef int mesh_project(A, mesh, spacing, d0, d1, n):
    cdef unsigned char[:,:,:] stack = A
    cdef double[:,:] verts = mesh.verts
    cdef double[:,:] norms = mesh.vert_props['normal']
    cdef double[:] signal = mesh.vert_props['signal']
    cdef double[:] sp = spacing

    _mesh_project_mean(&stack[0,0,0], stack.shape[0], stack.shape[1], stack.shape[2], &sp[0], &verts[0,0], &norms[0,0], &signal[0], verts.shape[0], d0, d1, n)
    return 0


