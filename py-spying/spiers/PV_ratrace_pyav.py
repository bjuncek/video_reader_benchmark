import os
import av

path = "../videos/RATRACE_wave_f_nm_np1_fr_goo_37.avi"

for i in range(1000):
    images_av = []
    container = av.open(path)
    container.streams.video[0].thread_type = 'AUTO'
    for frame in container.decode(video=0):
        images_av.append(frame)

print(len(images_av))
