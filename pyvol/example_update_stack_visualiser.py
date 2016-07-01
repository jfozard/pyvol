"""Example update stack volume visualiser.

press j and k to toggle between two stacks
"""

import sys
import logging
import types

import numpy as np

from renderer import BaseGlutWindow, VolumeRenderer

from io.tiff_parser import open_tiff

class ExampleVolumeVisualiser(BaseGlutWindow):

    def load_image(self, fpath, spacing):
        self.renderer = VolumeRenderer()
        self.stack = open_tiff(fpath)
        self.renderer.make_volume_obj(self.stack, spacing)

    def draw_hook(self):
        self.renderer.render(self.width, self.height, self.VMatrix, self.PMatrix)

    def reshape_hook(self):
        self.renderer.init_back_texture(self.width, self.height)
        pass


def original(self, x, y):
    self.renderer.volume_objects[0].update_stack(self.stack)


def inverted(self, x, y):
    stack = np.ones(self.stack.shape, np.uint8) * 255
    stack = stack - self.stack
    stack = stack * 0.1  # Make it a bit transparent.
    self.renderer.volume_objects[0].update_stack(stack)


def main():
    logging.basicConfig(level=logging.DEBUG)
    r = ExampleVolumeVisualiser("Example Volume Visualiser", 800, 600)
    if len(sys.argv) >= 5:
        spacing = map(float, sys.argv[2:5])
    else:
        spacing = (1.0, 1.0, 1.0)
    r.load_image(sys.argv[1], spacing)

    r.original = types.MethodType(original, r)
    r.inverted = types.MethodType(inverted, r)
    r.key_bindings["o"] = r.original
    r.key_bindings["i"] = r.inverted

    r.start()

if __name__ == '__main__':
    main()

