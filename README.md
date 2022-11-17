# hdbank


Create a template bank for nonprecessing binary using hybrid geometric-random template placement method.
For details of how this code works see:
*  S. Roy, A. S. Sengupta, and N. Thakor, Phys. Rev. D95, 104045 (2017), [1702.06771](https://arxiv.org/abs/1702.06771)
*  S. Roy, A. S. Sengupta,  and P. Ajith, Phys. Rev. D99, 024048 (2019), [1711.08743](https://arxiv.org/abs/1711.08743)




## Installation
One key component for generating the hybrid bank is metric over the dimensionless chirp time coordinate. To compute the metric, we modified in the LALSimulation code base. So, we first create a conda environment and install LALSuite.

### Simple installation
```
git clone https://github.com/soumenroy/hdbank.git
cd hdbank
sh install.sh
```

### Install step-by-step
1. Clone hdbank repository
  ```
  git clone https://github.com/soumenroy/hdbank.git
  cd hdbank
  ```
2. Clone LALSuite repository
```
git clone https://git.ligo.org/lscsoft/lalsuite.git
cd lalsuite
git checkout lalsuite-v7.11
cp ../ modify.patch
git apply modify.patch
```
3. Create conda environment and install LALSuite 
```
conda env create --name hdb -f common/conda/environment.yml
conda activate hdb
./00boot
./configure --prefix=$CONDA_PREFIX --enable-swig-python --disable-all-lal  --enable-lalsimulation --enable-lalinspiral --enable-lalframe --enable-lalmetaio --enable-lalburst
make -j 4 && make install
cd ..
```
4. Install hdbank
`pip install .`


## Structure of placement algorithm

<img src="https://github.com/soumenroy/hdbank/blob/cd05a491a80206ffd13c978a52b546cbe6c1507f/docs/algo.png" width="400">
