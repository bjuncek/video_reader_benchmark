#!/bin/bash

# py-spy record --native -o out/ratrace_cv2.svg -- python spiers/PV_ratrace_CV2.py
# py-spy record --native -o out/SOX5_cv2.svg -- python spiers/PV_SOX5_CV2.py

# py-spy record --native -o out/ratrace_av.svg -- python spiers/PV_ratrace_pyav.py
# py-spy record --native -o out/SOX5_av.svg -- python spiers/PV_SOX5_pyav.py


py-spy record --native -o out/ratrace_decord.svg -- python spiers/PV_ratrace_decord.py
py-spy record --native -o out/SOX5_decord.svg -- python spiers/PV_SOX5_decord.py
