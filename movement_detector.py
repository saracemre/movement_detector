# camera_movement.py
"""
Advanced camera movement detector.
Usage:
    from camera_movement import CameraMovementDetector, plot_movements
    det = CameraMovementDetector()
    moves = det.detect(frames)          # frames = list of BGR frames
    plot_movements(frames, moves)       # see results
"""

from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple


class CameraMovementDetector:
    def __init__(
        self,
        translation_thresh_px: float = 0.02,   # percentage of image diagonal
        rotation_thresh_deg: float = 1.0,
        scale_thresh: float = 0.05,
        min_good_matches: int = 20,
        ratio_test: float = 0.75,
        inlier_ratio_thresh: float = 0.30,
        orb_features: int = 2000,
    ):
        self.translation_thresh_px = translation_thresh_px
        self.rotation_thresh_deg = rotation_thresh_deg
        self.scale_thresh = scale_thresh
        self.min_good_matches = min_good_matches
        self.ratio_test = ratio_test
        self.inlier_ratio_thresh = inlier_ratio_thresh

        self.orb = cv2.ORB_create(nfeatures=orb_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # --------------------------------------------------------------------- #
    #  Helper functions
    # --------------------------------------------------------------------- #
    @staticmethod
    def _decompose_homography(H: np.ndarray) -> Dict[str, float]:
        """Homography -> tx, ty, rot (deg), sx, sy."""
        if H is None:
            return dict(tx=0, ty=0, rot=0, sx=1, sy=1)

        H = H / H[2, 2]
        a, b, c, d = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
        tx, ty = H[0, 2], H[1, 2]
        sx = np.sqrt(a * a + c * c)
        sy = np.sqrt(b * b + d * d)
        rot = np.degrees(np.arctan2(c, a))
        return dict(tx=tx, ty=ty, rot=rot, sx=sx, sy=sy)

    def _kp_desc(self, frame: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(gray, None)

    # --------------------------------------------------------------------- #
    #  Main function
    # --------------------------------------------------------------------- #
    def detect(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Returns camera movements (pan/tilt/zoom) between consecutive frames.
        Only frames with detected movement are included in the result list.
        """
        if len(frames) < 2:
            return []

        # 1) Pre-cache features
        feats = [self._kp_desc(f) for f in frames]

        # 2) Resolution-dependent threshold
        h, w = frames[0].shape[:2]
        diag = np.hypot(h, w)
        tx_thresh = self.translation_thresh_px * diag

        movements: List[Dict[str, Any]] = []

        # 3) Iterate over frame pairs
        for i in range(1, len(frames)):
            kp1, des1 = feats[i - 1]
            kp2, des2 = feats[i]
            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                continue

            # Ratio-test matching
            matches = self.bf.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < self.ratio_test * n.distance]
            if len(good) < self.min_good_matches:
                continue

            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                continue

            inlier_ratio = float(mask.mean())
            if inlier_ratio < self.inlier_ratio_thresh:
                # Most likely object motion in the scene
                continue

            dec = self._decompose_homography(H)
            t = float(np.hypot(dec["tx"], dec["ty"]))
            r = float(abs(dec["rot"]))
            s = float(max(abs(1 - dec["sx"]), abs(1 - dec["sy"])))

            reasons = []
            if t > tx_thresh:
                reasons.append("translation")
            if r > self.rotation_thresh_deg:
                reasons.append("rotation")
            if s > self.scale_thresh:
                reasons.append("scale")

            if reasons:
                movements.append(
                    dict(
                        frame_index=i,
                        reasons=reasons,
                        translation_px=t,
                        rotation_deg=r,
                        scale_diff=s,
                        inlier_ratio=inlier_ratio,
                        details=dec,
                    )
                )
        return movements


# ------------------------------------------------------------------------- #
#  Helper function for visualizing results
# ------------------------------------------------------------------------- #
def plot_movements(
    frames: List[np.ndarray],
    movements: List[Dict[str, Any]],
    cols: int = 5,
) -> None:
    """Shows frames with detected movement in red title."""
    idx2reason = {m["frame_index"]: ", ".join(m["reasons"]) for m in movements}
    n = len(frames)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
        title = f"{i}"
        if i in idx2reason:
            title += f"\n{idx2reason[i]}"
            ax.set_title(title, color="red")
        else:
            ax.set_title(title)
    for j in range(n, len(axes)):
        axes[j].remove()

    plt.tight_layout()
    plt.show()
