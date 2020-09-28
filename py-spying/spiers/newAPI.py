import os
import torch
import torchvision

print("inports over")
videos = [os.path.join("../videos/", x) for x in os.listdir("../videos") if x not in ["README", ".ipynb_checkpoints"]]

print("Now starting video decoding for profile")

for i in range(1000):
    for path in videos:
        reader = torch.classes.torchvision.Video(path, "video", True)
        frames = []
        t = reader.next_tensor("")
        while t.numel() > 0:
            frames.append(t)
            t = reader.next_tensor("")
        print(len(frames))