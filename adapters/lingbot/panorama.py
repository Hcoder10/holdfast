"""Rotational scene memory: remember what was seen at each camera orientation.

For a camera that only rotates, any earlier frame maps onto a later view by a
homography, with no depth needed: x_target = K R_target^T R_key K^-1 x_key.
We keep keyframes with their orientation and, for a requested orientation,
composite the remembered pixels into that view together with a validity mask.
The oldest keyframe that covers a pixel wins, so the first view of a place is
the one that persists and later re-renders cannot drift it.
"""
import math

import cv2
import numpy as np


def rot(yaw_deg, pitch_deg=0.0):
    """Camera-to-world rotation for a yaw (right positive) and pitch (up positive)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return ry @ rx


def angle_between(ra, rb):
    c = (np.trace(ra.T @ rb) - 1) / 2
    return math.degrees(math.acos(float(np.clip(c, -1, 1))))


class RotationMemory:
    def __init__(self, K, width, height, max_angle=65.0, min_spacing=6.0, border=6):
        self.K, self.Kinv = K, np.linalg.inv(K)
        self.w, self.h = width, height
        self.max_angle, self.min_spacing, self.border = max_angle, min_spacing, border
        self.keys = []  # (R c2w, frame uint8 HxWx3), oldest first

    def add(self, frame, yaw, pitch=0.0, force=False):
        r = rot(yaw, pitch)
        if not force and any(angle_between(r, k[0]) < self.min_spacing for k in self.keys):
            return False
        self.keys.append((r, frame.copy()))
        return True

    def render(self, yaw, pitch=0.0):
        """Remembered view at an orientation: (image uint8 HxWx3, mask bool HxW)."""
        rt = rot(yaw, pitch)
        img = np.zeros((self.h, self.w, 3), np.uint8)
        mask = np.zeros((self.h, self.w), bool)
        src_valid = np.zeros((self.h, self.w), np.uint8)
        b = self.border
        src_valid[b:self.h - b, b:self.w - b] = 255
        for r, frame in self.keys:  # oldest first: first view wins
            if angle_between(rt, r) > self.max_angle:
                continue
            H = self.K @ rt.T @ r @ self.Kinv
            warped = cv2.warpPerspective(frame, H, (self.w, self.h), flags=cv2.INTER_LINEAR)
            valid = cv2.warpPerspective(src_valid, H, (self.w, self.h), flags=cv2.INTER_NEAREST) > 0
            take = valid & ~mask
            img[take] = warped[take]
            mask |= take
            if mask.all():
                break
        return img, mask
