from setuptools import setup

readme = open('README.md').read()
version = "0.1.1"

setup(name="pyvol",
      packages=["pyvol", "pyvol.external", "pyvol.parser", "pyvol.mesh", "pyvol.shaders"],
      package_data={"pyvol.shaders": ["*.frag", "*.vert"]},
      version=version,
      description="Python package for creating custom 3D viewers",
      long_description=readme,
      license='MIT',
      keywords=["bioimage", "3D", "visualiser"],
      install_requires=[
        "numpy",
        "scipy",
        "Pillow",
        "PyOpenGL",
        "PyOpenGl-accelerate",
      ]
)
