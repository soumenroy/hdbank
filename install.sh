#!/bin/bash

git clone https://git.ligo.org/lscsoft/lalsuite.git
cp modify.patch lalsuite
cd lalsuite
git checkout lalsuite-v7.3

git apply modify.patch
conda env create --name hdb -f conda/environment.yml
eval "$(conda shell.bash hook)"
conda activate hdb
./00boot
./configure --prefix=$CONDA_PREFIX --enable-swig-python --disable-all-lal --enable-lalsimulation
make -j 4 && make install

cd ..
pip install .
