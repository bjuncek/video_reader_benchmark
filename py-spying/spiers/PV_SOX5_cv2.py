import os
import cv2


path = "../videos/SOX5yA1l24A.mp4"

for i in range(1000):
    images_cv2 = []
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images_cv2.append(frame)
        else:
            break
    cap.release()

print(len(images_cv2))