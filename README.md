# hdbank


Create a template bank for nonprecessing binary using hybrid geometric-random template placement method.
For details of how this code works see:
*  S. Roy, A. S. Sengupta, and N. Thakor, Phys. Rev. D95, 104045 (2017), [1702.06771](https://arxiv.org/abs/1702.06771)
*  S. Roy, A. S. Sengupta,  and P. Ajith, Phys. Rev. D99, 024048 (2019), [1711.08743](https://arxiv.org/abs/1711.08743)



## Structure of placement algorithm

[Algo]()


## Installation
One key component for generating the hybrid bank is metric over the dimensionless chirp time coordinate. To compute the metric, we modified in the LALSimulation code base. So, we first create a conda environment and install LALSuite.

### Simple installation
```
git clone https://github.com/soumenroy/hdbank.git
cd hdbank
sh install.sh
```

### Install step-by-step
#### Clone hdbank repository
```
git clone https://github.com/soumenroy/hdbank.git
cd hdbank
```
#### Clone LALSuite repository
```
git clone https://git.ligo.org/lscsoft/lalsuite.git
cd lalsuite
git checkout lalsuite-v7.3
cp ../ modify.patch
git apply modify.patch
```
#### Create conda environment and install LALSuite 
```
conda env create --name hdb -f conda/environment.yml
conda activate hdb
./00boot
./configure --prefix=$CONDA_PREFIX --enable-swig-python --enable-lalsimulation --enable-lalinspiral
make -j 4 && make install
cd ..
```
#### Install hdbank
`pip install .`


