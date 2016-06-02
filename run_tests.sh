#!/usr/bin/bash

source env/bin/activate
python pyvol/example_volume_visualiser.py test_data/wall-hypocotyl.tif
python pyvol/example_isosurface_visualiser.py test_data/wall-hypocotyl.tif
python pyvol/example_solid_visualiser.py test_data/sample_new.ply
