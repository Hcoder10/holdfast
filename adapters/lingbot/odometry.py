"""Visual odometry for rotation: how far did the camera actually turn?

Between two frames related by a (near) pure camera rotation, image points map
by the homography H = K R K^-1. We match features, fit H with RANSAC, recover R
and read off yaw and pitch. Used to measure achieved rotation against the
commanded one, and to close the loop on turns.
"""
import math

import cv2
import numpy as np


def intrinsics_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def relative_rotation(frame_a, frame_b, K, max_features=3000):
    """Rotation taking camera A's orientation to camera B's, as (yaw_deg,
    pitch_deg, inlier_count). Positive yaw = camera turned right, positive
    pitch = camera looked up. Returns (None, None, 0) if matching fails."""
    ga = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None, None, 0
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, None, 0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None:
        return None, None, 0
    # x_b = K R_ba K^-1 x_a, where R_ba maps directions from A's frame to B's.
    R = np.linalg.inv(K) @ H @ K
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        R = -R
    # Camera-to-world rotation of B relative to A is R^T; its forward axis:
    fwd = R.T @ np.array([0.0, 0.0, 1.0])
    yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
    pitch = math.degrees(math.atan2(-fwd[1], math.hypot(fwd[0], fwd[2])))
    return yaw, pitch, int(mask.sum())


def integrate(frames, K, stride=4):
    """Accumulate yaw/pitch along a clip by chaining frames `stride` apart.
    Returns a list of (frame_index, yaw_deg, pitch_deg, inliers)."""
    out, yaw, pitch = [(0, 0.0, 0.0, 0)], 0.0, 0.0
    for i in range(stride, len(frames), stride):
        dy, dp, n = relative_rotation(frames[i - stride], frames[i], K)
        if dy is not None:
            yaw, pitch = yaw + dy, pitch + dp
        out.append((i, yaw, pitch, n))
    return out
