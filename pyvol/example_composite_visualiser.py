"""Example composite visualiser."""

import sys
from renderer import BaseGlutWindow, CompositeRenderer

from imageio.tiff_parser import open_tiff

class ExampleCompositeVisualiser(BaseGlutWindow):

    def load_image(self, fpath, spacing):
        self.renderer = CompositeRenderer()
        stack = open_tiff(fpath)
        self.renderer.make_volume_obj(stack, spacing)

    def load_mesh(self, fpath, spacing):
        self.renderer.make_solid_obj(fpath, spacing)
        self.renderer.solid_objects[-1].transform = self.renderer.volume_objects[-1].transform

    def draw_hook(self):
        self.renderer.render(self.width, self.height, self.VMatrix, self.PMatrix)

    def reshape_hook(self):
        self.renderer.init_back_texture(self.width, self.height)

def main():
    r = ExampleCompositeVisualiser("Example Composite Visualiser", 800, 600)
    if len(sys.argv) >= 5:
        spacing = map(float, sys.argv[2:5])
    else:
        spacing = (1.0, 1.0, 1.0)
    r.load_image(sys.argv[1], spacing)
    r.load_mesh(sys.argv[-1], spacing)
    r.start()

if __name__ == '__main__':
    main()
