import logging

import numpy as np

from parser.ply_parser import parse_ply, write_ply
from algo import *


class Mesh(object):
    """
    Very simple triangulated mesh class
    Stored as triangle soup
    """

    def __init__(self):
        # List of vertex positions
        self.verts = np.zeros((0,3), dtype=float)
        # List of triangle vertex indices
        self.tris = np.zeros((0,3), dtype=int)
        # Dictionary of arrays - per vertex properties
        self.vert_props = {}
        # Dictionary of arrays - per face properties
        self.tri_props = {}

    def load_ply(self, fn):
        descr, data = parse_ply(fn)

        logging.debug( descr )

        NV = len(data['vertex'][0])
        NF = len(data['face'][0])
        logging.debug( 'NF {}'.format(NF) )

        verts = []

        logging.debug( data['vertex'][1] )
        x_idx = data['vertex'][1].index('x')
        y_idx = data['vertex'][1].index('y')
        z_idx = data['vertex'][1].index('z')

        # Does the surface have normal data?
        if 'nx' in data['vertex'][1]:
            nx_idx = data['vertex'][1].index('nx')
            ny_idx = data['vertex'][1].index('ny')
            nz_idx = data['vertex'][1].index('nz')
            has_normal = True
            vert_norm = []
        else:
            has_normal = False

        if 'signal' in data['vertex'][1]:
            s_idx = data['vertex'][1].index('signal')
            has_signal = True
            vert_signal = []
        else:
            has_signal = False

        if 'red' in data['vertex'][1]:
            r_idx = data['vertex'][1].index('red')
            b_idx = data['vertex'][1].index('blue')
            g_idx = data['vertex'][1].index('green')
            vert_color = []
            has_color = True
        else:
            has_color = False

        if 'label' in data['vertex'][1]:
            l_idx = data['vertex'][1].index('label')
            vert_label = []
            has_label = True
        elif 'state' in data['vertex'][1]:
            l_idx = data['vertex'][1].index('state')
            vert_label = []
            has_label = True
        else:
            has_label = False

        for v in data['vertex'][0]:
            verts.append((v[x_idx], v[y_idx], v[z_idx]))
            if has_normal:
                vert_norm.append(np.array((v[nx_idx], v[ny_idx], v[nz_idx])))
            if has_label:
                vert_label.append(int(v[l_idx]))
            if has_signal:
                vert_signal.append(v[s_idx])
            if has_color:
                vert_color.append(np.array((v[r_idx], v[g_idx], v[b_idx])))

        logging.debug( 'done_vertex' )

        self.verts = np.array(verts)

        tris = []
        for f in data['face'][0]:
            vv = f[0]
            tt = []
            for i in range(len(vv)-2):
                tt.append((vv[0], vv[i+1], vv[i+2]))

            tris.extend(tt)

        self.tris = np.array(tris)

        if has_normal:
            self.vert_props['normal'] = np.array(vert_norm)
        else:
            logging.debug( 'Calculate surface normals' )
            # Area weighted surface normals (would prefer angle-weighted)
            self.vert_props['normal'] = calculate_vertex_normals(self.verts, self.tris)

        if has_signal:
            self.vert_props['signal'] = np.array(vert_signal)

        if has_color:
            self.vert_props['color'] = np.array(vert_color)

        if has_label:
            self.vert_props['label'] = np.array(vert_label)


    def save_ply(self, filename):

        descr = [('vertex', None, [('x', ['float']),
                                   ('y', ['float']),
                                   ('z', ['float'])]),
                 ('face', None, [('vertex_index',
                                  ['list', 'int', 'int'])])]

        vp_list = ['x', 'y', 'z']
        fp_list = ['vertex_index']
        v_data = []
        f_data = []

        if 'normal' in self.vert_props:
            descr[0][2].extend([('nx', ['float']),
                                ('ny', ['float']),
                                ('nz', ['float'])])
            vp_list.extend(['nx', 'ny', 'nz'])
            normal = self.vert_props['normal']
            has_normal = True
        else:
            has_normal = False

        if 'color' in self.vert_props:
            descr[0][2].extend([('red', ['uchar']),
                                ('green', ['uchar']),
                                ('blue', ['uchar'])])
            vp_list.extend(['red', 'green', 'blue'])
            color = self.vert_props['color']
            has_color = True
        else:
            has_color = False

        if 'signal' in self.vert_props:
            descr[0][2].append(('signal', ['float']))
            vp_list.extend('signal')
            signal = self.vert_props['signal']
            has_signal = True
        else:
            has_signal = False

        if 'label' in self.vert_props:
            descr[0][2].append(('label', ['int']))
            vp_list.extend('label')
            label = self.vert_props['label']
            has_label = True
        else:
            has_label = False


        for i in range(self.verts.shape[0]):
            v = self.verts[i,:].tolist()
            if has_normal:
                v.extend(normal[i,:])
            if has_color:
                v.extend(color[i,:])
            if has_signal:
                v.append(signal[i])
            if has_label:
                v.append(label[i])
            v_data.append(v)

        for i in range(self.tris.shape[0]):
            t = self.tris[i,:].tolist()
            f_data.append([t])

        data = { 'vertex': (v_data, vp_list),
                 'face': (f_data, fp_list) }

        logging.debug( 'write_ply', filename )
        write_ply(filename, descr, data)
        logging.debug( 'write_ply_done' )






