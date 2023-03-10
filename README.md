# General
Benchmarking and profiling various video reading options for python


## Linux setup
Basically, constrained to a conda env

```
conda create --name vb python=3.10
conda activate vb

# usefull benchmarking tool (courtesy of Ralf)
pip install py-spy

# pytorch install (need GPU version for benchmark (arrrgh))
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia


# genral setup
pip install -r requirements.txt
conda install -y -c conda-forge jupyterlab

# first setting up pyav
conda install -y av -c conda-forge

# mixing pip and conda hurts me more than it should :|
pip install opencv-python 
pip install decord


### build torchvision from source to support video reader
conda install -y ffmpeg -c conda-forge
mkdir -p ./bin; cd ./bin
git clone git@github.com:pytorch/vision.git vision_nightly
cd vision_nightly
# remove old TV
pip uninstall torchvision; rm -rf build;  python setup.py install
# check if the installation is complete
python test/test_videoapi.py

```



## Notes 
### py-spying
This folder provides everything necessary to generate py-spy profiles for various readers; idea behind this, was to measure and compare the functional calls of various libraries; not that for this
to work everything needs to be built in DEBUG mode, which is a pain for FFMPEG. 

### timeitcomp
Compare runtimes of various libraries. Results are generated to the `out` folder, and `Graph Results` notebook can be used to visualize the results.
Files:
- `sequential_compare.py` - compare the lenght taken for a sequential read of a video
- `random_compare.py` - compare the time taken for a random read for k frames from a video (k={1,5,10}); Note that time to create a reader instance is _not_ included in the comparison, as this is a one time cost, compared to seeking. `decord` uses the `get_batch` api, instead of the `NextFrame()` api, which is more similar to the other libraries, but comparatively slower. We are sampling 3 clips from each video.

`Graph Results` notebook can be used to visualize the results: in cell 2, please select which file you want to visualise out of the options.

### torch_overhead
Measure overhead of various torch allocations.
