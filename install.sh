git clone https://github.com/soumenroy/hdbank.git

git clone https://git.ligo.org/lscsoft/lalsuite.git
cd lalsuite
git checkout lalsuite-v7.3
wget https://raw.githubusercontent.com/soumenroy/hdbank/main/patch/modify.patch
git apply modify.patch
conda env create --name hdb -f conda/environment.yml
conda activate hdb
./00boot
./configure --prefix=$CONDA_PREFIX --enable-swig-python --disable-all-lal --enable-lalsimulation
make -j 4 && make install

cd ..
pip install .
