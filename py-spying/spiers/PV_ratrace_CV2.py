import os
import cv2

path = "../videos/RATRACE_wave_f_nm_np1_fr_goo_37.avi"

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
