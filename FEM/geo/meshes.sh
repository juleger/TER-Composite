#!/bin/bash
# generate_meshes.sh
lc0=0.5
for i in {0..7}; do
    lc=$(echo "$lc0 / (2^$i)" | bc -l)
    lc_fmt=$(printf "%.8f" $lc)
    gmsh -2 -o ../mesh/square/square_Q1_lc$lc_fmt.msh -setnumber lc $lc rectangle.geo
done