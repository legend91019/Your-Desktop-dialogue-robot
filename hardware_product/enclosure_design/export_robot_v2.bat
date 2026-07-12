@echo off
setlocal

if not exist stl mkdir stl

openscad.com -o stl\xinbao_robot_front_shell_v2.stl -D PART_ID=1 xinbao_robot_enclosure_v2.scad
openscad.com -o stl\xinbao_robot_back_shell_v2.stl  -D PART_ID=2 xinbao_robot_enclosure_v2.scad
openscad.com -o stl\xinbao_robot_base_body_v2.stl   -D PART_ID=3 xinbao_robot_enclosure_v2.scad

echo Done. Robot-shaped v2 STL files are in the stl folder.
