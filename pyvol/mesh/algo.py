

import numpy as np
import scipy.linalg as la

def calculate_vertex_normals(verts, tris):
    v_array = np.array(verts)
    tri_array = np.array(tris, dtype=int)
    tri_pts = v_array[tri_array]
    n = np.cross( tri_pts[:,1] - tri_pts[:,0], 
                  tri_pts[:,2] - tri_pts[:,0])


    v_normals = np.zeros(v_array.shape)

    for i in range(tri_array.shape[0]):
        for j in tris[i]:
            v_normals[j,:] += n[i,:]

    nrms = np.sqrt(v_normals[:,0]**2 + v_normals[:,1]**2 + v_normals[:,2]**2)
    
    v_normals = v_normals / nrms.reshape((-1,1))

    return v_normals

        
