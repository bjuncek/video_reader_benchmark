import torchvision

path = "../videos/RATRACE_wave_f_nm_np1_fr_goo_37.avi"

for i in range(1000):
    frames = []
    reader = torchvision.io.VideoReader(path, "video")
    for data in reader:
        frames.append(data["data"])

print(len(frames))
