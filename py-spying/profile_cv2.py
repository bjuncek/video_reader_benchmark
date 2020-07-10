import os
import cv2

print("inports over")
videos = [os.path.join("../videos/", x) for x in os.listdir("../videos") if x not in ["README", ".ipynb_checkpoints"]]

print("Now starting video decoding for profile")

for path in videos:
    images_cv2 = []
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images_cv2.append(frame)
        else:
            break
    cap.release()