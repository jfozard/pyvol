"""Example volume visualiser."""

import sys
import logging

from renderer import BaseGlutWindow, VolumeRenderer

from io.tiff_parser import open_tiff

class ExampleVolumeVisualiser(BaseGlutWindow):

    def load_image(self, fpath, spacing):
        self.renderer = VolumeRenderer()
        stack = open_tiff(fpath)
        self.renderer.make_volume_obj(stack, spacing)

    def draw_hook(self):
        self.renderer.render(self.width, self.height, self.VMatrix, self.PMatrix)

    def reshape_hook(self):
        self.renderer.init_back_texture(self.width, self.height)
        pass

def main():
    logging.basicConfig(level=logging.DEBUG)
    r = ExampleVolumeVisualiser("Example Volume Visualiser", 800, 600)
    if len(sys.argv) >= 5:
        spacing = map(float, sys.argv[2:5])
    else:
        spacing = (1.0, 1.0, 1.0)
    r.load_image(sys.argv[1], spacing)
    r.start()

if __name__ == '__main__':
    main()
