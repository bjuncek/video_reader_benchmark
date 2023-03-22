import torchvision
import torch
import decord
import av 
import cv2
import datetime, sys, platform

import subprocess

ffmpeg_version = subprocess.check_output(["ffmpeg", "-version"]).decode("utf-8")[15:20]

print("___ System versions ___")
print(f"Today's date: {datetime.datetime.now()}")
print(f"System: {platform.system()}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")
print(f"Python version: {sys.version}")
print("System ffmpeg version: ", ffmpeg_version)
print("___ Library versions ___")
print("OpenCV version: ", cv2.__version__)
print(f"pytorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"decord version: {decord.__version__}")
print(f"pyav version: {av.__version__}")
