"""Measure achieved camera rotation in a video: python measure_yaw.py video.mp4 [fx] [stride]

Prints cumulative yaw/pitch every 12 frames (one generation step) so it can be
compared with what was commanded. fx is for an 832-wide frame (default 503).
"""
import os
import sys

import imageio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapters", "lingbot"))
from odometry import integrate, intrinsics_matrix  # noqa: E402

frames = imageio.mimread(sys.argv[1], memtest=False)
fx = float(sys.argv[2]) if len(sys.argv) > 2 else 503.0
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 4
h, w = frames[0].shape[:2]
K = intrinsics_matrix(fx * w / 832, fx * w / 832, w / 2, h / 2)
track = integrate(frames, K, stride)
for i, yaw, pitch, n in track:
    if i % 12 == 0 or i == track[-1][0]:
        print(f"frame {i:4d} t={i/16:5.2f}s yaw {yaw:7.1f} pitch {pitch:6.1f} inliers {n}")
