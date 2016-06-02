"""Example solid visualiser."""

import sys
from renderer import BaseGlutWindow, SolidRenderer

class ExampleSolidVisualiser(BaseGlutWindow):

    def load_image(self, fpath, spacing):
        self.renderer = SolidRenderer()
        self.renderer.make_solid_obj(fpath, spacing)

    def draw_hook(self):
        self.renderer.render(self.width, self.height, self.VMatrix, self.PMatrix)

    def reshape_hook(self):
        #self.renderer.init_back_texture(self.width, self.height)
        pass

def main():
    r = ExampleSolidVisualiser("Example Solid Visualiser", 800, 600)
    if len(sys.argv) >= 5:
        spacing = map(float, sys.argv[2:5])
    else:
        spacing = (1.0, 1.0, 1.0)
    r.load_image(sys.argv[1], spacing)
    r.start()

if __name__ == '__main__':
    main()
