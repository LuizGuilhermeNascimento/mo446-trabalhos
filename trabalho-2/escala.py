import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

import charuco

ROI_MARGIN_MM, ROI_HEIGHT_MM = 30, 250


def triangulate(projections, rays):
    A = np.concatenate([x[:, None] * P[2] - P[:2] for P, x in zip(projections, rays)])
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3]


def reprojection_errors(track, X):
    return np.array([np.linalg.norm(image.camera.img_from_cam(image.cam_from_world() * X) - uv) for image, uv in track])


def triangulate_track(track, max_error=2.0):
    for _ in range(2):
        X = triangulate([image.cam_from_world().matrix() for image, _ in track],
                        [image.camera.cam_from_img(uv) for image, uv in track])
        errors = reprojection_errors(track, X)
        track = [obs for obs, e in zip(track, errors) if e < max_error]
        if len(track) < 3:
            return None, None
    return X, reprojection_errors(track, X)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = (S * D.diagonal()).sum() / (a ** 2).sum()
    return s, R, mu_d - s * R @ mu_s


def board_points(data):
    path = data / "medidas.json"
    square = json.loads(path.read_text()).get("quadrado_mm", charuco.SQUARE_MM) if path.exists() else charuco.SQUARE_MM
    return charuco.BOARD.getChessboardCorners().astype(float) * [1, -1, 0] * square / charuco.SQUARE_MM, square


def board_region(data, margin_mm):
    board, square = board_points(data)
    return board[:, :2].min(0) - square - margin_mm, board[:, :2].max(0) + square + margin_mm


def corner_tracks(rec, data):
    tracks = {}
    for image in rec.images.values():
        if image.has_pose:
            # COLMAP ignores the EXIF orientation, so the raw pixels are read here too.
            corners, ids = charuco.detect(cv2.imread(str(data / image.name), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION))
            for i, uv in zip(ids, corners.astype(float) + 0.5):
                tracks.setdefault(i, []).append((image, uv))
    return tracks


def main(data, out):
    rec = pycolmap.Reconstruction(out / "esparso")
    board, square = board_points(data)
    model, target, errors, views = [], [], [], []
    for i, track in corner_tracks(rec, data).items():
        X, e = triangulate_track(track) if len(track) >= 3 else (None, None)
        if X is not None:
            model.append(X), target.append(board[i]), errors.extend(e), views.append(len(e))
    model, target = np.array(model), np.array(target)
    if len(model) < 6:
        sys.exit(f"apenas {len(model)} cantos do tabuleiro triangulados")

    keep = np.ones(len(model), bool)
    for _ in range(3):
        s, R, t = umeyama(model[keep], target[keep])
        residual = np.linalg.norm(s * model @ R.T + t - target, axis=1)
        keep = residual < max(3 * np.median(residual[keep]), 0.5)
    rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
    if np.median([im.projection_center()[2] for im in rec.images.values() if im.has_pose]) <= 0:
        sys.exit("câmeras abaixo do tabuleiro: alinhamento inválido")

    (out / "metrico").mkdir(exist_ok=True)
    rec.write(out / "metrico")
    lo, hi = board_region(data, ROI_MARGIN_MM)
    roi = rec.crop(pycolmap.AlignedBox3d(np.r_[lo, -ROI_MARGIN_MM], np.r_[hi, ROI_HEIGHT_MM]))
    (out / "roi").mkdir(exist_ok=True)
    roi.write(out / "roi")
    shutil.rmtree(out / "mvs", ignore_errors=True)
    pycolmap.undistort_images(out / "mvs", out / "roi", data)

    report = {
        "quadrado_mm": square,
        "mm_por_unidade": s,
        "cantos_usados": int(keep.sum()),
        "cantos_descartados": int((~keep).sum()),
        "vistas_por_canto_mediana": float(np.median(views)),
        "residuo_alinhamento_mm": {"rms": float(np.sqrt((residual[keep] ** 2).mean())),
                                   "mediana": float(np.median(residual[keep])), "max": float(residual[keep].max())},
        "reprojecao_cantos_px": {"media": float(np.mean(errors)), "mediana": float(np.median(errors))},
        "pontos_esparsos_roi": roi.num_points3D(),
        "sim3": {"s": s, "R": R.tolist(), "t": t.tolist()},
    }
    (out / "escala.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sim3"}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
