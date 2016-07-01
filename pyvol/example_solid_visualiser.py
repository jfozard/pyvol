"""Example solid visualiser.

press 't' to toggle solid on and off
"""

import sys
import logging
import types

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


def toggle_solid(self, x, y):
    self.renderer.solid_objects[0].active = not self.renderer.solid_objects[0].active


def main():
    logging.basicConfig(level=logging.DEBUG)
    r = ExampleSolidVisualiser("Example Solid Visualiser", 800, 600)
    if len(sys.argv) >= 5:
        spacing = map(float, sys.argv[2:5])
    else:
        spacing = (1.0, 1.0, 1.0)
    r.load_image(sys.argv[1], spacing)

    r.toggle_solid = types.MethodType(toggle_solid, r)
    r.key_bindings["t"] = r.toggle_solid

    r.start()

if __name__ == '__main__':
    main()
