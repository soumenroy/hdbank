#!/bin/bash

git clone https://git.ligo.org/lscsoft/lalsuite.git
cp modify.patch lalsuite
cd lalsuite
git checkout lalsuite-v7.3

git apply modify.patch
conda env create --name hdb -f conda/environment.yml

# https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
eval "$(conda shell.bash hook)"

conda activate hdb
./00boot
./configure --prefix=$CONDA_PREFIX --enable-swig-python --disable-all-lal  --enable-lalsimulation --enable-lalinspiral --enable-lalframe --enable-lalmetaio --enable-lalburst
make -j 4 && make install

cd ..
pip install .
