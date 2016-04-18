"""Module for parsing tiff files."""

import numpy as np
from PIL import Image

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


