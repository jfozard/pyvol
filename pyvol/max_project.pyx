cimport cython
from cython.parallel import parallel, prange


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int max_project(unsigned char[:,:,:] A, double[:,:] t, double[:,:] r) nogil:
    cdef size_t i, j, k
    cdef size_t I, J, K
    cdef float c
    I = A.shape[0]
    J = A.shape[1]
    K = A.shape[2]
    for j in prange(J):
        for k in range(K):
            c = t[j,k]
            for i in range(I):
                if (<float> A[i,j,k]) > c:
                    r[j,k] = <float> i
                    break
                r[j,k] = I - 1.0
    return 0    
