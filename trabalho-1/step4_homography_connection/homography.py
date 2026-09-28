from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from common import IMAGE_TEXT, IMAGE_TITLE, save_figure
from step1_key_points_detection.detector import Features
from step2_feature_matching.matcher import pair_canvas, sample

RANSAC_THRESHOLD = 3.0  # px; OpenCV's default ransacReprojThreshold
RANSAC_MAX_ITERS = 2000  # OpenCV's default maxIters
RANSAC_CONFIDENCE = 0.995  # OpenCV's default confidence

INLIER_COLOR = "lime"  # RANSAC inliers (filled dots + lines)
OUTLIER_COLOR = "red"  # RANSAC outliers (x markers)


@dataclass
class PairHomography:
    """RANSAC homography between images i and j, estimated from the ratio-test matches."""

    i: int
    j: int
    H: np.ndarray | None  # 3x3, maps image i -> image j; None when estimation failed
    pts_i: np.ndarray  # Nx2 matched points in image i
    pts_j: np.ndarray  # Nx2 matched points in image j
    inlier_mask: np.ndarray  # N booleans

    @property
    def n_matches(self) -> int:
        return len(self.pts_i)

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum())

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_matches if self.n_matches else 0.0

    @property
    def mean_reproj_error(self) -> float:
        """Mean distance (px) between H(p_i) and p_j over the inliers."""
        if self.H is None or self.n_inliers == 0:
            return float("nan")
        return float(reprojection_errors(self.H, self.pts_i[self.inlier_mask], self.pts_j[self.inlier_mask]).mean())

    def inverse(self) -> PairHomography:
        """Same pair seen from j to i."""
        H = None if self.H is None else np.linalg.inv(self.H)
        return PairHomography(self.j, self.i, H, self.pts_j, self.pts_i, self.inlier_mask)


def reprojection_errors(H: np.ndarray, pts_i: np.ndarray, pts_j: np.ndarray) -> np.ndarray:
    projected = cv2.perspectiveTransform(pts_i.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    return np.linalg.norm(projected - pts_j, axis=1)


def signed_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_plausible(H: np.ndarray, pts_i: np.ndarray) -> bool:
    """Rejects degenerate homographies: the bounding box of the matched points of image i must map
    to a convex quadrilateral with the same orientation (no mirroring) and in front of the camera."""
    (x0, y0), (x1, y1) = pts_i.min(axis=0), pts_i.max(axis=0)
    box = np.float64([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    w = np.c_[box, np.ones(4)] @ H[2]
    if np.any(w <= 0):  # a corner crosses the horizon line of the homography
        return False
    quad = cv2.perspectiveTransform(box.reshape(-1, 1, 2), H).reshape(-1, 2)
    return cv2.isContourConvex(quad.astype(np.float32)) and signed_area(quad) * signed_area(box) > 0


def estimate_pair_homography(i: int, j: int, feat_i: Features, feat_j: Features,
                             matches: list[cv2.DMatch]) -> PairHomography:
    """findHomography with RANSAC on the ratio-test matches of the pair (requirement 5.1)."""
    pts_i = feat_i.points[[m.queryIdx for m in matches]].reshape(-1, 2)
    pts_j = feat_j.points[[m.trainIdx for m in matches]].reshape(-1, 2)
    return fit_homography(i, j, pts_i, pts_j)


def fit_homography(i: int, j: int, pts_i: np.ndarray, pts_j: np.ndarray) -> PairHomography:
    """RANSAC homography from matched points (also used after the cylindrical projection)."""
    no_model = PairHomography(i, j, None, pts_i, pts_j, np.zeros(len(pts_i), bool))
    if len(pts_i) < 4:
        return no_model
    H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, RANSAC_THRESHOLD,
                                 maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE)
    if H is None or not is_plausible(H, pts_i):
        return no_model
    return PairHomography(i, j, H, pts_i, pts_j, mask.ravel().astype(bool))


def plot_ransac(img_i: np.ndarray, img_j: np.ndarray, pair: PairHomography, title: str, path: Path) -> None:
    """As in Figure 5 of the assignment: inliers = filled dots (joined by lines), outliers = x."""
    canvas, offset = pair_canvas(img_i, img_j)
    h, w = canvas.shape[:2]
    fig, ax = plt.subplots(figsize=(14, 14 * h / w + 1.0))
    ax.imshow(canvas)
    inliers = sample(list(np.flatnonzero(pair.inlier_mask)))
    outliers = np.flatnonzero(~pair.inlier_mask)
    pi, pj = pair.pts_i, pair.pts_j + [offset, 0]
    if inliers:
        ax.add_collection(LineCollection(np.stack([pi[inliers], pj[inliers]], axis=1), colors=INLIER_COLOR,
                                         linewidths=0.8, alpha=0.8))
        for pts in (pi[inliers], pj[inliers]):
            ax.scatter(pts[:, 0], pts[:, 1], s=14, c=INLIER_COLOR, edgecolors="black", linewidths=0.3, zorder=3)
    for pts in (pi[outliers], pj[outliers]):
        ax.scatter(pts[:, 0], pts[:, 1], s=28, c=OUTLIER_COLOR, marker="x", linewidths=1.2, zorder=3)
    ax.legend(handles=[Line2D([], [], color=INLIER_COLOR, marker="o", mec="black", ls="-", label="inlier"),
                       Line2D([], [], color=OUTLIER_COLOR, marker="x", ls="", label="outlier")],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncols=2, fontsize=IMAGE_TEXT, frameon=False)
    ax.set_title(title, fontsize=IMAGE_TITLE)
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, path)
