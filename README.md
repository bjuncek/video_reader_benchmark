# General
Benchmarking and profiling various video reading options for python


## Setup on a mac - PROBABLY DEPRECATED
Basically, constrained to a conda env

```
conda create --name vb python=3.7
conda activate vb

# usefull benchmarking tool (courtesy of Ralf)
pip install py-spy

# pytorch install (on a mac, locally)
conda install -y pytorch torchvision -c pytorch

# genral setup
conda install -y -c conda-forge jupyterlab
conda install -y seaborn matplotlib pandas

# first setting up pyav
conda install -y av -c conda-forge
# then install opencv
pip install opencv-python 

# then decord
pip install decord


# TODO: install torchvision from source to support video reader
mkdir -p ~/bin; cd bin
git clone git@github.com:pytorch/vision.git
cd vision
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install

```


## Notes and descriptions

### py-spying
This folder provides everything necessary to generate py-spy profiles for various readers; idea behind this, was to measure and compare the functional calls of various libraries. 

### timeitcomp
Compare runtimes of various libraries. Results are generated to the `out` folder, and `Graph Results` notebook can be used to visualize the results

### torch_overhead
Measure overhead of various torch allocations.
