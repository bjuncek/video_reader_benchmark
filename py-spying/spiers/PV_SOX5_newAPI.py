import torchvision

path = "../videos/SOX5yA1l24A.mp4"

for i in range(1000):
    frames = []
    reader = torchvision.io.VideoReader(path, "video")
    for data in reader:
        frames.append(data["data"])

print(len(frames))
