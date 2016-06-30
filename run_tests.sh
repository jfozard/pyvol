#!/usr/bin/bash

source env/bin/activate
python pyvol/example_volume_visualiser.py test_data/wall-hypocotyl.tif 0.293 0.293 0.444
python pyvol/example_isosurface_visualiser.py test_data/wall-hypocotyl.tif
python pyvol/example_solid_visualiser.py test_data/hypercotyl.ply
python pyvol/example_composite_visualiser.py test_data/wall-hypocotyl.tif 0.293 0.293 0.444 test_data/hypercotyl.ply
