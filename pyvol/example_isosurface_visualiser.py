"""Example isosurface visualiser.

press 't' to toggle isosurface on and off
"""

import sys
import logging
import types

from renderer import BaseGlutWindow, IsosurfaceVolumeRenderer

from parser.tiff_parser import open_tiff

class ExampleIsosurfaceVisualiser(BaseGlutWindow):

    def load_image(self, fpath, spacing):
        self.renderer = IsosurfaceVolumeRenderer()
        stack = open_tiff(fpath)
        self.renderer.make_volume_obj(stack, spacing)

    def draw_hook(self):
        self.renderer.render(self.width, self.height, self.VMatrix, self.PMatrix)

    def reshape_hook(self):
        self.renderer.init_back_texture(self.width, self.height)
        pass


def toggle_volume(self, x, y):
    self.renderer.volume_objects[0].active = not self.renderer.volume_objects[0].active


def main():
    logging.basicConfig(level=logging.DEBUG)
    r = ExampleIsosurfaceVisualiser("Example Isosurface Visualiser", 800, 600)
    if len(sys.argv) >= 5:
        spacing = map(float, sys.argv[2:5])
    else:
        spacing = (1.0, 1.0, 1.0)
    r.load_image(sys.argv[1], spacing)

    r.toggle_volume = types.MethodType(toggle_volume, r)
    r.key_bindings["t"] = r.toggle_volume

    r.start()

if __name__ == '__main__':
    main()
