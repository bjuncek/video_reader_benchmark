import os
import cv2

path = "../videos/SOX5yA1l24A.mp4"

for i in range(1000):
    images_cv2 = []
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            images_cv2.append(frame)
        else:
            break
    cap.release()

print(len(images_cv2))
