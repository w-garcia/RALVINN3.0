#!/bin/bash
# Verify that anaconda is installed
conda list $
echo Installing Theano... $
conda install --channel https://conda.anaconda.org/Toli theano $
echo Installing pygame... $
conda install --channel https://conda.anaconda.org/kne pygame $
echo Installing opencv3... $
conda install --channel https://conda.anaconda.org/menpo opencv3 $
echo Installing lasagne... $
conda install --channel https://conda.anaconda.org/Toli lasagne $
echo Removing bundled libgfortran package... $
conda remove libgfortran... $
echo Installing rgrout libgfortran package... $
conda install --channel https://conda.anaconda.org/rgrout libgfortran $
echo Removing bundled scipy package... $
conda remove scipy $
echo Installing nanshe scipy package with openblas $
conda install --channel https://conda.anaconda.org/nanshe scipy $
echo Installing Kivy $
conda install --channel https://conda.anaconda.org/wgarcia kivy $
echo Installing SDL2 implement of pygame $
conda install --channel https://conda.anaconda.org/kne pygame_sdl2 $
echo Ready.
