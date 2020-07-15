#!/bin/bash

py-spy record --native -o out/cv2.svg -- python spiers/profile_cv2.py
py-spy record --native -o out/cv2_long.svg -- python spiers/profile_cv2_long.py

py-spy record --native -o out/pyav.svg -- python spiers/profile_av.py
py-spy record --native -o out/pyav_long.svg -- python spiers/profile_av_long.py

py-spy record --native -o out/tvvr.svg -- python spiers/profile_tvvr.py
py-spy record --native -o out/tvvr_long.svg -- python spiers/profile_tvvr_long.py

py-spy record --native -o out/decord.svg -- python spiers/profile_decord.py
py-spy record --native -o out/decord_long.svg -- python spiers/profile_decord_long.py