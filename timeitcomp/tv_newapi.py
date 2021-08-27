import argparse
import timeit
import os
import pandas as pd

import itertools
import torchvision

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("n", type=int, help="Number of trials to run")
args = parser.parse_args()


setup_newAPI = """\
import torch
import torchvision
"""


def measure_reading_video(path):
    """We pass a path and get the
    loaded the video
    """
    frames = []
    reader = torchvision.io.VideoReader(path, "video")
    for data in reader:
        frames.append(data["data"])


def measure_seek_and_decode(reader, seek_in_s, num_frames=10):
    frames = []
    for data in itertools.islice(reader.seek(seek_in_s), num_frames):
        frames.append(data["data"])


loaders = []
times_per_video = []
times_random_seek = []
video = []
num_frames = []
lib_version = []


for i in range(args.n):
    print(i)
    for file in os.listdir("../videos"):
        if file in ["README", ".ipynb_checkpoints", "avadl.py"]:
            print(f"Skipping {file}")
            continue

        path = os.path.join("../videos/", file)

        frames = []
        reader = torchvision.io.VideoReader(path, "video")
        for data in reader:
            frames.append(data["data"])
        nframes = len(frames)

        md = reader.get_metadata()
        duration = md["video"]["duration"][0]
        time = duration / 2

        print(path, nframes, duration)

        video.append(file)
        loaders.append("VideoReader")
        num_frames.append(nframes)
        lib_version.append(torchvision.__version__)

        times_per_video.append(
            timeit.timeit(
                f'measure_reading_video("{path}")',
                setup=setup_newAPI,
                globals=globals(),
                number=args.n,
            )
            / args.n
        )

        times_random_seek.append(
            timeit.timeit(
                f"measure_seek_and_decode(reader, time)",
                setup=setup_newAPI,
                globals=globals(),
                number=args.n,
            )
            / args.n
        )


df = pd.DataFrame(
    {
        "decoder": loaders,
        "video": video,
        "time": times_per_video,
        "num_frames": num_frames,
        "lib_version": lib_version,
    }
)
df.to_csv("out/tvNewAPI_fullvideo.csv")

df = pd.DataFrame(
    {
        "decoder": loaders,
        "video": video,
        "time": times_random_seek,
        "num_frames": num_frames,
        "lib_version": lib_version,
    }
)
df.to_csv("out/tvNewAPI_random_seek.csv")
