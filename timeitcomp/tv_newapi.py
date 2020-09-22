import argparse
import timeit
import os
import pandas as pd

import torch
import torchvision

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('n', type=int,
                    help='Number of trials to run')
args = parser.parse_args()


setup_newAPI = """\
import torch
import torchvision
"""


def next_list(path):
    reader = torch.classes.torchvision.Video(path, "video", True)
    frames = []
    t, _ = reader.next_list("")
    while t.numel() > 0:
        frames.append(t)
        t, _ = reader.next_list("")

def next_tensor(path):
    reader = torch.classes.torchvision.Video(path, "video", True)
    frames = []
    t = reader.next_tensor("")
    while t.numel() > 0:
        frames.append(t)
        t = reader.next_tensor("")


def next_list_dummy_tensor(path):
    reader = torch.classes.torchvision.Video(path, "video", True)
    frames = []
    t, _ = reader.next_list_dummy_tensor("")
    while t.numel() > 0:
        frames.append(t)
        t, _ = reader.next_list_dummy_tensor("")

def next_tensor_dummy_tensor(path):
    reader = torch.classes.torchvision.Video(path, "video", True)
    frames = []
    t = reader.next_tensor_dummy_tensor("")
    while t.numel() > 0:
        frames.append(t)
        t = reader.next_tensor_dummy_tensor("")


def next_tensor_usemove(path):
    reader = torch.classes.torchvision.Video(path, "video", True)
    frames = []
    t = reader.next_tensor_usemove("")
    while t.numel() > 0:
        frames.append(t)
        t = reader.next_tensor_usemove("")

def next_list_usemove(path):
    reader = torch.classes.torchvision.Video(path, "video", True)
    frames = []
    t, _ = reader.next_list_usemove("")
    while t.numel() > 0:
        frames.append(t)
        t, _ = reader.next_list_usemove("")


def next_int_numframes(path):
    reader = torch.classes.torchvision.Video(path, "video", True)
    i = reader.next_int_numframes("")
    frames = []
    while i == 1:
        i = reader.next_int_numframes("")
        frames.append(i)
    

def fullvideo_numframes(path):
    reader = torch.classes.torchvision.Video(path, "video", True)
    _ = reader.fullvideo_numframes()


def fullvideo_tensor(path):
    reader = torch.classes.torchvision.Video(path, "video", True)
    _ = reader.fullvideo_tensor()

        
loaders = []
times = []
video = []
num_frames = []
lib_version = []


for i in range(args.n):
    print(i)
    for file in os.listdir("../videos"):
        if file in ["README", ".ipynb_checkpoints"]:
            print("Skipping README")
            continue
          
        path = os.path.join("../videos/", file)
        print(path)
        reader = torch.classes.torchvision.Video(path, "video", True)
        nframes = reader.fullvideo_numframes()
        
        times.append( timeit.timeit(f"next_list(\"{path}\")", setup=setup_newAPI, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("next_list")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)

        times.append( timeit.timeit(f"next_tensor(\"{path}\")", setup=setup_newAPI, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("next_tensor")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)

        times.append( timeit.timeit(f"next_int_numframes(\"{path}\")", setup=setup_newAPI, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("next_int_numframes")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)

        times.append( timeit.timeit(f"next_list_usemove(\"{path}\")", setup=setup_newAPI, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("next_list_usemove")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)

        times.append( timeit.timeit(f"next_list_dummy_tensor(\"{path}\")", setup=setup_newAPI, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("next_list_dummy_tensor")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)

    
        times.append( timeit.timeit(f"next_tensor_dummy_tensor(\"{path}\")", setup=setup_newAPI, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("next_tensor_dummy_tensor")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)

        times.append( timeit.timeit(f"next_tensor_usemove(\"{path}\")", setup=setup_newAPI, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("next_tensor_usemove")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)


        times.append( timeit.timeit(f"fullvideo_numframes(\"{path}\")", setup=setup_newAPI, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("fullvideo_numframes")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)

        times.append( timeit.timeit(f"fullvideo_tensor(\"{path}\")", setup=setup_newAPI, globals=globals(), number=args.n)/args.n)
        video.append(file)
        loaders.append("fullvideo_tensor")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)

        
df = pd.DataFrame({"loader": loaders, "video": video, "time":times, "num_frames":num_frames, "lib_version": lib_version})
df.to_csv("out/tv_newAPI.csv") 
        
        

