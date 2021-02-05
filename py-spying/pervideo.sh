#!/bin/bash

py-spy record --native -o out/ratrace_newAPI.svg -- python spiers/PV_ratrace_newAPI.py
py-spy record --native -o out/SOX5_newAPI.svg -- python spiers/PV_SOX5_newAPI.py

py-spy record --native -o out/ratrace_decord.svg -- python spiers/PV_ratrace_decord.py
py-spy record --native -o out/SOX5_decord.svg -- python spiers/PV_SOX5_decord.py

py-spy record --native -o out/SOX5_cv2.svg -- python spiers/PV_SOX5_cv2.py
py-spy record --native -o out/ratrace2_cv2.svg -- python spiers/PV_ratrace_cv2.py
