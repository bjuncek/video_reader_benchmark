from decord import VideoReader
from decord import cpu, gpu


path = "../videos/RATRACE_wave_f_nm_np1_fr_goo_37.avi"

for i in range(1000):
    images_d = []
    vr = VideoReader(path, ctx=cpu(0))
    for i in range(len(vr)):
        # the video reader will handle seeking and skipping in the most efficient manner
        images_d.append(vr[i])

print(len(images_d))