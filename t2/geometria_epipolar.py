"""Feature comparison, matching, and two-view epipolar geometry for T2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


METHODS = ("SIFT", "ORB", "AKAZE")


@dataclass
class Features:
    path: Path
    image: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None


def detect_features(
    image_path: str | Path,
    method: str,
    max_features: int = 3000,
) -> Features:
    """Read one image and compute SIFT, ORB, or AKAZE features."""
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    method = method.upper()
    if method == "SIFT":
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=max_features)
    elif method == "AKAZE":
        if not hasattr(cv2, "AKAZE_create"):
            raise RuntimeError(
                f"This OpenCV build ({cv2.__version__}) does not expose AKAZE_create. "
                "Install the T2 dependencies from requirements.txt in the active kernel."
            )
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unsupported descriptor: {method}")

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if len(keypoints) > max_features:
        selected = np.argsort([point.response for point in keypoints])[-max_features:]
        keypoints = [keypoints[index] for index in selected]
        descriptors = descriptors[selected]
    return Features(path, image, keypoints, descriptors)


def _norm_for(method: str) -> int:
    return cv2.NORM_L2 if method.upper() == "SIFT" else cv2.NORM_HAMMING


def _points(features1: Features, features2: Features, matches: list[cv2.DMatch]):
    p1 = np.float32([features1.keypoints[m.queryIdx].pt for m in matches])
    p2 = np.float32([features2.keypoints[m.trainIdx].pt for m in matches])
    return p1, p2


def _sampson_median(fundamental: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.median(_sampson_distances(fundamental, p1, p2)))


def _sampson_distances(fundamental: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([p1, np.ones(len(p1))])
    x2 = np.column_stack([p2, np.ones(len(p2))])
    fx1 = (fundamental @ x1.T).T
    ftx2 = (fundamental.T @ x2.T).T
    numerator = np.sum(x2 * fx1, axis=1) ** 2
    denominator = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    distance = np.sqrt(numerator / np.maximum(denominator, 1e-12))
    return distance


def _inlier_coverage(p1: np.ndarray, p2: np.ndarray, image1: np.ndarray, image2: np.ndarray) -> float | None:
    """Return the smaller convex-hull coverage of the inliers in the two frames."""
    if len(p1) < 3 or len(p2) < 3:
        return None
    ratios = []
    for points, image in ((p1, image1), (p2, image2)):
        hull = cv2.convexHull(points.astype(np.float32))
        ratios.append(float(cv2.contourArea(hull) / (image.shape[0] * image.shape[1])))
    return min(ratios)


def match_pair(
    features1: Features,
    features2: Features,
    method: str,
    *,
    ratio: float = 0.75,
    ransac_threshold_px: float = 1.5,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    cross_check: bool = False,
    confidence: float = 0.999,
) -> dict[str, Any]:
    """BF/Lowe matching, F with RANSAC, and optionally E/pose/triangulation."""
    if features1.descriptors is None or features2.descriptors is None:
        return {
            "matches": [], "match_count": 0, "ratio_match_count": 0,
            "fundamental": None, "fundamental_inliers": 0,
            "fundamental_inlier_ratio": 0.0, "median_sampson_px": None,
            "mean_sampson_px": None, "p90_sampson_px": None, "inlier_coverage": None,
            "median_inlier_displacement_px": None, "essential": None,
            "essential_inliers": 0, "rotation": None, "translation": None,
            "camera_center_2": None, "pose_inliers": 0,
            "triangulated_points": None, "triangulated_count": 0,
            "cross_check": cross_check, "reason": "no descriptors",
        }

    matcher = cv2.BFMatcher(_norm_for(method), crossCheck=False)
    neighbors = matcher.knnMatch(features1.descriptors, features2.descriptors, k=2)

    def ratio_matches(knn_matches: list[tuple[cv2.DMatch, ...]]) -> list[cv2.DMatch]:
        return [
            pair[0]
            for pair in knn_matches
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
        ]

    good = ratio_matches(neighbors)
    ratio_match_count = len(good)
    if cross_check:
        reverse_neighbors = matcher.knnMatch(
            features2.descriptors, features1.descriptors, k=2
        )
        reverse_pairs = {
            (match.trainIdx, match.queryIdx)
            for match in ratio_matches(reverse_neighbors)
        }
        good = [match for match in good if (match.queryIdx, match.trainIdx) in reverse_pairs]
    result: dict[str, Any] = {
        "matches": good,
        "match_count": len(good),
        "ratio_match_count": ratio_match_count,
        "cross_check": cross_check,
        "fundamental": None,
        "fundamental_inliers": 0,
        "fundamental_inlier_ratio": 0.0,
        "median_sampson_px": None,
        "mean_sampson_px": None,
        "p90_sampson_px": None,
        "inlier_coverage": None,
        "median_inlier_displacement_px": None,
        "essential": None,
        "essential_inliers": 0,
        "rotation": None,
        "translation": None,
        "camera_center_2": None,
        "triangulated_points": None,
        "triangulated_count": 0,
    }
    if len(good) < 8:
        result["reason"] = "fewer than 8 ratio-test matches"
        return result

    p1, p2 = _points(features1, features2, good)
    fundamental, f_mask = cv2.findFundamentalMat(
        p1,
        p2,
        cv2.FM_RANSAC,
        ransac_threshold_px,
        confidence,
        5000,
    )
    if fundamental is None or f_mask is None or fundamental.shape != (3, 3):
        result["reason"] = "fundamental matrix estimation failed"
        return result

    f_mask = f_mask.ravel().astype(bool)
    p1_inliers, p2_inliers = p1[f_mask], p2[f_mask]
    count = int(f_mask.sum())
    sampson_distances = _sampson_distances(fundamental, p1_inliers, p2_inliers) if count else np.array([])
    result.update(
        fundamental=fundamental,
        fundamental_mask=f_mask,
        fundamental_inliers=count,
        fundamental_inlier_ratio=count / len(good),
        median_sampson_px=float(np.median(sampson_distances)) if count else None,
        mean_sampson_px=float(np.mean(sampson_distances)) if count else None,
        p90_sampson_px=float(np.percentile(sampson_distances, 90)) if count else None,
        inlier_coverage=_inlier_coverage(
            p1_inliers, p2_inliers, features1.image, features2.image
        ),
        median_inlier_displacement_px=float(
            np.median(np.linalg.norm(p1_inliers - p2_inliers, axis=1))
        ) if count else None,
    )

    if camera_matrix is None or count < 5:
        result["reason"] = "camera intrinsics required for E, pose, and triangulation"
        return result

    normalized1_all = cv2.undistortPoints(
        p1_inliers.reshape(-1, 1, 2), camera_matrix, dist_coeffs
    ).reshape(-1, 2)
    normalized2_all = cv2.undistortPoints(
        p2_inliers.reshape(-1, 1, 2), camera_matrix, dist_coeffs
    ).reshape(-1, 2)
    focal_scale = float((camera_matrix[0, 0] + camera_matrix[1, 1]) / 2)
    normalized_threshold = ransac_threshold_px / focal_scale
    essential, e_mask = cv2.findEssentialMat(
        normalized1_all,
        normalized2_all,
        np.eye(3),
        method=cv2.RANSAC,
        prob=confidence,
        threshold=normalized_threshold,
    )
    if essential is None or e_mask is None:
        result["reason"] = "essential matrix estimation failed"
        return result

    essential_inliers = int(e_mask.ravel().astype(bool).sum())
    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        normalized1_all,
        normalized2_all,
        np.eye(3),
        mask=e_mask,
    )
    if pose_mask is None:
        pose_mask = np.zeros((len(normalized1_all), 1), dtype=np.uint8)
    pose_inliers = pose_mask.ravel().astype(bool)
    normalized1 = normalized1_all[pose_inliers]
    normalized2 = normalized2_all[pose_inliers]
    if len(normalized1) == 0:
        result.update(
            essential=essential,
            essential_inliers=essential_inliers,
            rotation=rotation,
            translation=translation,
            camera_center_2=(-rotation.T @ translation).ravel(),
            pose_inliers=0,
            triangulated_points=np.empty((0, 3), dtype=np.float64),
            triangulated_count=0,
            reason="recoverPose não encontrou inliers de pose; triangulação ignorada.",
        )
        return result

    projection1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    projection2 = np.hstack([rotation, translation])
    points4d = cv2.triangulatePoints(
        projection1,
        projection2,
        normalized1.reshape(-1, 2).T,
        normalized2.reshape(-1, 2).T,
    )
    if points4d is None or points4d.ndim != 2 or points4d.shape[0] != 4:
        result.update(
            essential=essential,
            essential_inliers=essential_inliers,
            rotation=rotation,
            translation=translation,
            camera_center_2=(-rotation.T @ translation).ravel(),
            pose_inliers=int(pose_inliers.sum()),
            triangulated_points=np.empty((0, 3), dtype=np.float64),
            triangulated_count=0,
            reason="triangulatePoints não retornou pontos homogêneos válidos.",
        )
        return result

    homogeneous_w = points4d[3]
    finite_w = np.abs(homogeneous_w) > 1e-12
    points3d = np.full((3, points4d.shape[1]), np.nan, dtype=np.float64)
    points3d[:, finite_w] = points4d[:3, finite_w] / homogeneous_w[finite_w]
    positive_depth = (points3d[2] > 0) & ((rotation @ points3d + translation)[2] > 0)
    triangulated_points = points3d[:, positive_depth].T
    result.update(
        essential=essential,
        essential_inliers=essential_inliers,
        rotation=rotation,
        translation=translation,
        camera_center_2=(-rotation.T @ translation).ravel(),
        pose_inliers=int(pose_inliers.sum()),
        triangulated_points=triangulated_points,
        triangulated_count=len(triangulated_points),
        reason=None,
    )
    return result


def analyze_sequence(
    image_paths: list[str | Path],
    *,
    methods: tuple[str, ...] = METHODS,
    max_features: int = 3000,
    max_gap: int = 2,
    ratio: float = 0.75,
    ransac_threshold_px: float = 1.5,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    cross_check: bool = False,
) -> dict[str, Any]:
    """Compare descriptors on the same neighboring pairs in a video sequence."""
    paths = [Path(path) for path in image_paths]
    if len(paths) < 2:
        raise ValueError("At least two images are needed.")
    if max_gap < 1:
        raise ValueError("max_gap must be at least 1.")

    output: dict[str, Any] = {"methods": {}, "image_count": len(paths)}
    pairs = [
        (i, j)
        for i in range(len(paths))
        for j in range(i + 1, min(len(paths), i + max_gap + 1))
    ]
    output["pair_count"] = len(pairs)

    for method in methods:
        features = [detect_features(path, method, max_features) for path in paths]
        pair_results = []
        for i, j in pairs:
            result = match_pair(
                features[i],
                features[j],
                method,
                ratio=ratio,
                ransac_threshold_px=ransac_threshold_px,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                cross_check=cross_check,
            )
            result.update(i=i, j=j)
            pair_results.append(result)
        valid = [row for row in pair_results if row["fundamental"] is not None]
        output["methods"][method] = {
            "features": features,
            "pairs": pair_results,
            "summary": {
                "mean_keypoints": float(np.mean([len(f.keypoints) for f in features])),
                "median_ratio_matches": float(np.median([p["ratio_match_count"] for p in pair_results])),
                "median_matches": float(np.median([p["match_count"] for p in pair_results])),
                "median_f_inliers": float(np.median([p["fundamental_inliers"] for p in pair_results])),
                "valid_f_pairs": len(valid),
                "median_f_inlier_ratio": float(np.median([p["fundamental_inlier_ratio"] for p in valid]))
                if valid else 0.0,
                "median_sampson_px": float(np.median([p["median_sampson_px"] for p in valid]))
                if valid else None,
                "median_inlier_displacement_px": float(
                    np.median([p["median_inlier_displacement_px"] for p in valid])
                ) if valid else None,
                "valid_essential_pairs": sum(p["essential"] is not None for p in pair_results),
                "median_essential_inliers": float(np.median([
                    p["essential_inliers"] for p in pair_results if p["essential"] is not None
                ])) if any(p["essential"] is not None for p in pair_results) else None,
                "median_pose_inliers": float(np.median([
                    p["pose_inliers"] for p in pair_results if p["rotation"] is not None
                ])) if any(p["rotation"] is not None for p in pair_results) else None,
                "median_triangulated_points": float(np.median([
                    p["triangulated_count"] for p in pair_results if p["rotation"] is not None
                ])) if any(p["rotation"] is not None for p in pair_results) else None,
            },
        }
    return output


def draw_inlier_matches(
    analysis: dict[str, Any],
    method: str,
    pair: tuple[int, int] | None = None,
    max_matches: int = 80,
) -> np.ndarray:
    """Render the best pair's RANSAC inliers (or a selected pair)."""
    method_result = analysis["methods"][method]
    candidates = [p for p in method_result["pairs"] if p["fundamental"] is not None]
    if not candidates:
        raise ValueError(f"No valid fundamental-matrix pair for {method}.")
    pair = (
        next((p for p in candidates if (p["i"], p["j"]) == pair), None)
        if pair is not None
        else None
    )
    pair = pair or max(candidates, key=lambda p: p["fundamental_inliers"])
    features = method_result["features"]
    good = pair["matches"]
    inlier_matches = [match for match, keep in zip(good, pair["fundamental_mask"]) if keep]
    if len(inlier_matches) > max_matches:
        chosen = np.linspace(0, len(inlier_matches) - 1, max_matches).astype(int)
        inlier_matches = [inlier_matches[k] for k in chosen]
    image = cv2.drawMatches(
        features[pair["i"]].image,
        features[pair["i"]].keypoints,
        features[pair["j"]].image,
        features[pair["j"]].keypoints,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return image


def draw_epipolar_lines(
    analysis: dict[str, Any],
    method: str,
    pair: tuple[int, int] | None = None,
    max_lines: int = 30,
) -> np.ndarray:
    """Draw F-derived epipolar lines for inlier correspondences in a selected pair."""
    rows = [row for row in analysis["methods"][method]["pairs"] if row["fundamental"] is not None]
    if not rows:
        raise ValueError(f"No valid fundamental-matrix pair for {method}.")
    row = next((entry for entry in rows if (entry["i"], entry["j"]) == pair), None) if pair else None
    row = row or max(rows, key=lambda entry: entry["fundamental_inliers"])
    features = analysis["methods"][method]["features"]
    left = features[row["i"]].image.copy()
    right = features[row["j"]].image.copy()
    matches = [m for m, keep in zip(row["matches"], row["fundamental_mask"]) if keep]
    if len(matches) > max_lines:
        indices = np.linspace(0, len(matches) - 1, max_lines).astype(int)
        matches = [matches[index] for index in indices]
    p1, p2 = _points(features[row["i"]], features[row["j"]], matches)
    lines_right = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, row["fundamental"])
    lines_left = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, row["fundamental"])
    rng = np.random.default_rng(0)
    height, width = left.shape[:2]

    def paint(image: np.ndarray, lines: np.ndarray, points: np.ndarray) -> None:
        for line, point, color in zip(lines.reshape(-1, 3), points, rng.integers(40, 255, (len(points), 3))):
            a, b, c = line
            if abs(b) > 1e-9:
                x0, y0 = 0, int(round(-c / b))
                x1, y1 = width - 1, int(round(-(c + a * (width - 1)) / b))
                cv2.line(image, (x0, y0), (x1, y1), tuple(map(int, color)), 1, cv2.LINE_AA)
            x, y = np.round(point).astype(int)
            cv2.circle(image, (x, y), 4, tuple(map(int, color)), -1, cv2.LINE_AA)

    paint(right, lines_right, p2)
    paint(left, lines_left, p1)
    if left.shape[0] != right.shape[0]:
        right = cv2.resize(right, (round(right.shape[1] * height / right.shape[0]), height))
    return np.hstack([left, right])
