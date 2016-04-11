
import numpy as np
cimport numpy as np

np.import_array()

cdef extern void img_make_iso(unsigned char* img, int NZ, int NY, int NX, unsigned char level, float** verts, int** tris, int* NV, int* NT)

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef data_to_np_int(int* data, int m, int n):
    cdef np.npy_intp dims[2]
    dims[0] = m
    dims[1] = n
    cdef np.ndarray[np.int32_t, ndim=2] arr = \
        np.PyArray_SimpleNewFromData(2, dims, np.NPY_INT32, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr

cdef data_to_np_float(float* data, int m, int n):
    cdef np.npy_intp dims[2]
    dims[0] = m
    dims[1] = n
    cdef np.ndarray[np.float32_t, ndim=2] arr = \
        np.PyArray_SimpleNewFromData(2, dims, np.NPY_FLOAT32, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr


def make_iso(unsigned char[:,:,:] img, unsigned char level):
    cdef size_t I, J, K
    I = img.shape[0]
    J = img.shape[1]
    K = img.shape[2]
    cdef float* verts
    cdef int* tris
    cdef int NV
    cdef int NT
    img_make_iso(&img[0,0,0], I, J, K, level, &verts, &tris, &NV, &NT)
    print NV, NT
    vbuf = data_to_np_float(verts, NV, 3)
    tbuf = data_to_np_int(tris, NT, 3)

    return (vbuf, tbuf)
            
            
