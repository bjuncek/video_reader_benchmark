import os
import av

print("inports over")
videos = [os.path.join("../videos/", x) for x in os.listdir("../videos") if x not in ["README", ".ipynb_checkpoints"]]

print("Now starting video decoding for profile")

for path in videos:
    images_av = []
    container = av.open(path)
    container.streams.video[0].thread_type = 'AUTO'
    for frame in container.decode(video=0):
        images_av.append(frame.to_rgb().to_ndarray())