import argparse
import os
import pandas as pd
import random

import torchvision
import torchaudio
import torch.utils.benchmark as benchmark

import itertools
import av
import math
from decord import VideoReader, cpu



parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--n", default=4, type=int, help="Number of trials to run")
parser.add_argument("--num_clips", default=3, type=int, help="Number of clips to sample")

args = parser.parse_args()


setup_decord = """\
from decord import VideoReader, cpu  #, gpu
"""

setup_newAPI = """\
import torch
import torchvision
torchvision.set_video_backend("video_reader")
"""

setup_tvvr = """\
import torch
import torchvision
torchvision.set_video_backend("video_reader")
"""

setup_tvpyav = """\
import torch
import torchvision
torchvision.set_video_backend("pyav")
"""

setup_cv2 = """\
import torch
import cv2
cv2.setNumThreads(1)
import numpy as np
"""

setup_pyav = """\
import torch
import av
import numpy as np
"""

setup_ta = """\
from torchaudio.io import StreamReader
"""


def get_pyav_random_seek(container, times_in_s, num_frames=10):
    stream = container.streams.video[0]
    stream_name = {"video": 0}
    results = []
    for time_in_s in times_in_s:
        start_offset = int(math.floor(time_in_s * (1 / stream.time_base)))
        container.seek(start_offset, any_frame=False, backward=True, stream=stream)

        frames = {}
        for _idx, frame in enumerate(container.decode(**stream_name)):
            if frame.pts > start_offset:
                frames[frame.pts] = frame

            if len(frames) >= num_frames:
                break

        result = [frames[i] for i in sorted(frames)]
        result = [frame.to_rgb().to_ndarray() for frame in result]
        results.append(result)
    reflen = len(results[0])
    for r in results:
        assert len(r) == reflen


def measure_seek_and_decode(reader, times_in_s, num_frames=10):
    results = []
    for time_in_s in times_in_s:
        frames = []
        for data in itertools.islice(reader.seek(time_in_s), num_frames):
            frames.append(data["data"])
        results.append(frames)
    reflen = len(results[0])
    for r in results:
        assert len(r) == reflen

def measure_decord_one(reader, times_in_s, num_frames=10):
    results = []
    for time_in_s in times_in_s:
        start_frm = int(time_in_s * reader.get_avg_fps())
        frames = reader.get_batch(list(range(start_frm, start_frm + num_frames))).asnumpy()
        results.append(frames)
    reflen = len(results[0])
    for r in results:
        assert len(r) == reflen


def measure_TA(vid, times_in_s, num_frames=10):
    results = []
    vid.add_basic_video_stream(
        frames_per_chunk=1,
        format="rgb24"
    )
    fps = vid.get_out_stream_info(0).frame_rate
    for time_in_s in times_in_s:
        vid.seek(0)
        curr, counter = 0, 0
        frames = []
        for chunks in vid.stream():
            if len(frames) < num_frames:
                curr = counter / fps
                if curr >= time_in_s:
                    frames.append(chunks[0])
                counter += 1
            else:
                break
        results.append(frames)    
    reflen = len(results[0])
    for r in results:
        assert len(r) == reflen


loaders = []
times = []
mean_times = []
video = []
num_frames = []
threads = []
codecs = []
_codecs = ["original", "h264", "xvid"]
for codec in _codecs:
    for num_threads in [1, 4, 8]:
        for file in os.listdir(f"../videos/{codec}"):
            if file in ["README", ".ipynb_checkpoints", "avadl.py"]:
                print(f"Skipping {file}")
                continue

            path = os.path.join(f"../videos/{codec}", file)
            images_av = []
            container = av.open(path)
            container.streams.video[0].thread_count = 1  # force single thread
            for frame in container.decode(video=0):
                images_av.append(frame.to_rgb().to_ndarray())
            nframes = len(images_av)

            container = av.open(path)
            duration = float(
                container.streams.video[0].duration * container.streams.video[0].time_base
            )
            try:
                seektimes = random.sample(range(0, int(duration-1)), args.num_clips)
            except ValueError:
                seektimes = [0, duration // 2]
            
            for frames_to_read in [1, 5, 10]:
                print(path, nframes, duration, seektimes, frames_to_read, codec)
                container = av.open(path)
                container.streams.video[0].thread_count = num_threads
                video.append(file)
                loaders.append("pyav")
                num_frames.append(frames_to_read)
                times.append(benchmark.Timer(
                        stmt=f"get_pyav_random_seek(container, seektimes, frames_to_read)",
                        setup=setup_pyav,
                        globals=globals(),
                        label="Video Reading",
                        sub_label=str(file),
                        description="pyav",
                        num_threads=num_threads,
                    ).timeit(args.n))
                mean_times.append(times[-1].mean)
                threads.append(num_threads)
                codecs.append(codec)

                torchvision.set_video_backend("video_reader")
                reader = torchvision.io.VideoReader(path, "video", num_threads=num_threads)
                video.append(file)
                loaders.append("tv_newapi")
                num_frames.append(frames_to_read)
                times.append(benchmark.Timer(
                        stmt=f"measure_seek_and_decode(reader, seektimes, frames_to_read)",
                        setup=setup_tvvr,
                        globals=globals(),
                        label="Video Reading",
                        sub_label=str(file),
                        description="tv_newapi",
                        num_threads=num_threads,
                    ).timeit(args.n))
                mean_times.append(times[-1].mean)
                threads.append(num_threads)
                codecs.append(codec)

                vr = VideoReader(path, ctx=cpu(0))
                video.append(file)
                loaders.append("decord")
                num_frames.append(frames_to_read)
                times.append(benchmark.Timer(
                        stmt=f"measure_decord_one(vr, seektimes, frames_to_read)",
                        setup=setup_decord,
                        globals=globals(),
                        label="Video Reading",
                        sub_label=str(file),
                        description="decord",
                        num_threads=num_threads,
                    ).timeit(args.n))
                mean_times.append(times[-1].mean)
                threads.append(num_threads)
                codecs.append(codec)

                vr = torchaudio.io.StreamReader(path)
                video.append(file)
                loaders.append("torchaudio")
                num_frames.append(frames_to_read)
                times.append(benchmark.Timer(
                        stmt=f"measure_TA(vr, seektimes, frames_to_read)",
                        setup=setup_decord,
                        globals=globals(),
                        label="Video Reading",
                        sub_label=str(file),
                        description="TORCHAUDIO",
                        num_threads=num_threads,
                    ).timeit(args.n))
                mean_times.append(times[-1].mean)
                threads.append(num_threads)
                codecs.append(codec)


compare = benchmark.Compare(times)
compare.print()
df = pd.DataFrame(
    {
        "decoder": loaders,
        "video": video,
        "time": mean_times,
        "num_frames": num_frames,
        "num_threads": threads,
        "codec": codecs,
    }
)
df.to_csv("out/READ_RANDOM_SEEK.csv")
