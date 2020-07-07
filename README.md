# General
Benchmarking and profiling various video reading options for python


## Setup
Basically, constrained to a conda env

```
# usefull benchmarking tool (courtesy of Ralf)
pip install py-spy

# pytorch install (on a mac, locally)
conda install pytorch torchvision -c pytorch

# TODO: install torchvision from source to support video reader

# genral setup
conda install -c conda-forge jupyterlab
conda install -y pandas, seaborn

# first setting up pyav
conda install av -c conda-forge
# then install opencv
pip install opencv-python 
```
