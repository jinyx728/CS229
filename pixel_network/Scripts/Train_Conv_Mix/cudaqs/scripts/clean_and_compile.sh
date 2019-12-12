#!/bin/bash
######################################################################
# Copyright 2019. Dan Johnson.
# This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
######################################################################
function clean_and_compile {
    rm -r lib_build
    mkdir lib_build
    cd lib_build
    cmake ..
    make
    cd ..
}

#echo "Delete build directory and recompile?"
#select yn in "Yes" "No"; do
#    case $yn in
#        Yes ) clean_and_compile; break;;
#        No ) exit;;
#    esac
#done

clean_and_compile
python setup.py install
