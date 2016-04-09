
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
    Extension("mesh_reproject",
              sources=["mesh_reproject.pyx", "_mesh_reproject.cpp"],
              extra_compile_args=['-O3', '-march=native', '--fast-math', '-fopenmp'],
              libraries=["m"], # Unix-like specific
              extra_link_args=['-fopenmp'],
    ),



]

setup(
  name = "mesh_reproject",
  ext_modules = cythonize(ext_modules)
)
