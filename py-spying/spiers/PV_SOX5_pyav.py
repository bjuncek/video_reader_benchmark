import os
import av

path = "../videos/SOX5yA1l24A.mp4"
print(os.path.exists(path))

for i in range(1000):
    images_av = []
    container = av.open(path)
    container.streams.video[0].thread_type = 'AUTO'
    for frame in container.decode(video=0):
        images_av.append(frame.to_rgb().to_ndarray())

print(len(images_av))