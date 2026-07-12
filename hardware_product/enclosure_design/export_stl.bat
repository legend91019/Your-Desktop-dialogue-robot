@echo off
setlocal

if not exist stl mkdir stl

openscad.com -o stl\xinbao_front_shell_v1.stl -D PART_ID=1 xinbao_enclosure_v0.scad
openscad.com -o stl\xinbao_back_shell_v1.stl  -D PART_ID=2 xinbao_enclosure_v0.scad
openscad.com -o stl\xinbao_base_v1.stl        -D PART_ID=3 xinbao_enclosure_v0.scad

echo Done. STL files are in the stl folder.
