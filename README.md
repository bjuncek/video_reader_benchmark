# General
Benchmarking and profiling various video reading options for python


## Linux setup
Basically, constrained to a conda env

```
conda create --name vb python=3.7
conda activate vb

# usefull benchmarking tool (courtesy of Ralf)
pip install py-spy

# pytorch install (latest CPUonly)
conda install pytorch torchvision cpuonly -c pytorch-nightly -y

# genral setup
conda install -y -c conda-forge jupyterlab
conda install -y seaborn matplotlib pandas pytest cmake make

# first setting up pyav
conda install -y av -c conda-forge

# mixing pip and conda hurts me more than it should :|
# then install opencv
pip install opencv-python 
# then decord
pip install decord


### build torchvision from source to support video reader
conda install -y ffmpeg -c conda-forge
mkdir -p ~/bin; cd ~/bin
git clone git@github.com:pytorch/vision.git vision_nightly
cd vision_nightly
# remove old TV
pip uninstall torchvision
python setup.py install
# check if the installation is complete
python test/test_videoapi.py

```




## Notes 
### py-spying
This folder provides everything necessary to generate py-spy profiles for various readers; idea behind this, was to measure and compare the functional calls of various libraries; not that for this
to work everything needs to be built in DEBUG mode, which is a pain for FFMPEG. 

### timeitcomp
Compare runtimes of various libraries. Results are generated to the `out` folder, and `Graph Results` notebook can be used to visualize the results

### torch_overhead
Measure overhead of various torch allocations.
