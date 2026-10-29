import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

import charuco

ROI_MARGIN_MM, ROI_HEIGHT_MM = 30, 250
PLANE_TOL_MM, RULER_RADIUS_MM = 1.5, 150


def triangulate(projections, rays):
    A = np.concatenate([x[:, None] * P[2] - P[:2] for P, x in zip(projections, rays)])
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3]


def reprojection_errors(track, X):
    return np.array([np.linalg.norm(image.camera.img_from_cam(image.cam_from_world() * X) - uv) for image, uv in track])


def triangulate_track(track, max_error=2.0):
    if len(track) < 3:
        return None, None
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


def measurements(data):
    path = data / "medidas.json"
    return json.loads(path.read_text()) if path.exists() else {}


def board_points(data):
    square = measurements(data).get("quadrado_mm", charuco.SQUARE_MM)
    return charuco.BOARD.getChessboardCorners().astype(float) * [1, -1, 0] * square / charuco.SQUARE_MM, square


def corner_tracks(rec, data):
    tracks = {}
    for image in rec.images.values():
        if image.has_pose:
            # COLMAP ignores the EXIF orientation, so the raw pixels are read here too.
            corners, ids = charuco.detect(cv2.imread(str(data / image.name), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION))
            for i, uv in zip(ids, corners.astype(float) + 0.5):
                tracks.setdefault(i, []).append((image, uv))
    return tracks


def board_frame(rec, data):
    board, square = board_points(data)
    model, target, errors, views = [], [], [], []
    for i, track in corner_tracks(rec, data).items():
        X, e = triangulate_track(track) if len(track) >= 3 else (None, None)
        if X is not None:
            model.append(X), target.append(board[i]), errors.extend(e), views.append(len(e))
    model, target = np.array(model), np.array(target)
    if len(model) < 6:
        sys.exit(f"apenas {len(model)} cantos do tabuleiro triangulados")

    (s, R, t), fit = robust_similarity(model, target)
    area = (board[:, :2].min(0) - square, board[:, :2].max(0) + square)
    return (s, R, t), area, {
        "quadrado_mm": square,
        "cantos_usados": fit.pop("usados"),
        "cantos_descartados": fit.pop("descartados"),
        "vistas_por_canto_mediana": float(np.median(views)),
        **fit,
        "reprojecao_cantos_px": {"media": float(np.mean(errors)), "mediana": float(np.median(errors))},
    }


def robust_similarity(model, target):
    keep = np.ones(len(model), bool)
    for _ in range(3):
        s, R, t = umeyama(model[keep], target[keep])
        residual = np.linalg.norm(s * model @ R.T + t - target, axis=1)
        keep = residual < max(3 * np.median(residual[keep]), 0.5)
    return (s, R, t), {"usados": int(keep.sum()), "descartados": int((~keep).sum()),
                       "residuo_alinhamento_mm": {"rms": float(np.sqrt((residual[keep] ** 2).mean())),
                                                  "mediana": float(np.median(residual[keep])), "max": float(residual[keep].max())}}


def plane_points_frame(rec, data):
    # Points clicked on a flat printed pattern (e.g. grid paper) with known table coordinates: {"x,y": {"<image>": [u, v]}}.
    images = {im.name: im for im in rec.images.values() if im.has_pose}
    model, target, errors = [], [], []
    for xy, clicks in measurements(data)["pontos_plano"].items():
        X, e = triangulate_track([(images[n], np.asarray(uv, float) + 0.5) for n, uv in clicks.items() if n in images], 4.0)
        if X is not None:
            model.append(X), target.append([*map(float, xy.split(",")), 0.0]), errors.extend(e)
    model, target = np.array(model), np.array(target)
    if len(model) < 4:
        sys.exit(f"apenas {len(model)} pontos do plano triangulados")
    (s, R, t), fit = robust_similarity(model, target)
    center = s * look_at_point(*camera_axes(rec)) @ R.T + t
    area = (center[:2] - RULER_RADIUS_MM, center[:2] + RULER_RADIUS_MM)
    return (s, R, t), area, {"pontos_usados": fit.pop("usados"), "pontos_descartados": fit.pop("descartados"), **fit,
                             "reprojecao_pontos_px": {"media": float(np.mean(errors)), "mediana": float(np.median(errors))}}


def orthophoto(image, data, center, x, y, half, n=1024):
    a = np.linspace(-half, half, n)
    P = center + a[None, :, None] * x + a[:, None, None] * y
    X = image.cam_from_world() * P.reshape(-1, 3)
    uv = image.camera.img_from_cam(X).astype(np.float32).reshape(n, n, 2) - 0.5
    gray = cv2.imread(str(data / image.name), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    ortho = cv2.remap(gray, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
    return ortho, (X[:, 2].reshape(n, n) > 0) & (ortho > 0)


def grid_period(ortho, valid, pixel):
    # Dominant spatial frequency of the printed grid; returns (period, angle, fraction of the view that is paper).
    # Keep only the paper (the brightest region), so the object's own texture does not compete with the grid.
    paper = (valid & (ortho > 0.8 * np.percentile(ortho[valid], 95))).astype(np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    paper = cv2.erode(cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((31, 31), np.uint8)), np.ones((9, 9), np.uint8)) > 0
    f = np.where(paper, ortho - cv2.GaussianBlur(ortho, (0, 0), 8), 0)
    F = np.abs(np.fft.fftshift(np.fft.fft2(f * np.outer(np.hanning(len(f)), np.hanning(len(f))))))
    n, c = len(F), len(F) // 2
    r = np.hypot(*np.mgrid[-c:n - c, -c:n - c])
    F[(r < 12) | (r > n / 6)] = 0  # ignore the paper's large-scale shading and pixel-level noise
    j, i = np.unravel_index(np.argmax(F), F.shape)
    # Thin lines have strong harmonics: if half the frequency also has a clear peak, that is the fundamental.
    while True:
        hj, hi = c + round((j - c) / 2), c + round((i - c) / 2)
        sub = F[hj - 1:hj + 2, hi - 1:hi + 2]
        if np.hypot(hj - c, hi - c) < 12 or sub.max() < 0.25 * F[j, i]:
            break
        dj, di = np.unravel_index(np.argmax(sub), sub.shape)
        j, i = hj - 1 + dj, hi - 1 + di
    # The FFT bin is ~1/(number of visible periods) wide; refine on a continuous grid of frequencies around it.
    yy, xx = np.nonzero(paper[::2, ::2])
    vals = f[::2, ::2][yy, xx]
    k0 = np.array([i - c, j - c], float)
    for step in (0.5, 0.1, 0.02):
        offs = np.arange(-5, 6) * step
        best = max(((dx, dy) for dx in offs for dy in offs),
                   key=lambda d: abs(np.exp(-2j * np.pi * ((k0[0] + d[0]) * 2 * xx + (k0[1] + d[1]) * 2 * yy) / n) @ vals))
        k0 += best
    return n * pixel / np.linalg.norm(k0), np.arctan2(k0[1], k0[0]), float(paper.mean())


def grid_frame(rec, data):
    # Grid paper of known spacing under the object: the period seen in orthophotos of the table plane gives the scale.
    spacing = measurements(data)["grade_mm"]
    centers, axes = camera_axes(rec)
    target = look_at_point(centers, axes)
    reach = np.median(np.linalg.norm(centers - target, axis=1))
    points = np.array([p.xyz for p in rec.points3D.values()])
    origin, z, inliers = fit_plane(points[np.linalg.norm(points - target, axis=1) < reach], 0.003 * reach, orbit_up(centers), centers)
    if np.mean((centers - origin) @ z) < 0:
        z = -z
    center = target - ((target - origin) @ z) * z
    x = np.cross(z, [0, 0, 1] if abs(z[2]) < 0.9 else [1, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    images = [im for im in rec.images.values() if im.has_pose]
    # Near-vertical views: in oblique ones a small error of the estimated plane changes the orthophoto scale more.
    down = sorted(images, key=lambda im: im.cam_from_world().rotation.matrix()[2] @ z)
    top = [im for im in down if im.cam_from_world().rotation.matrix()[2] @ z < -np.cos(np.radians(25))] or down[:4]
    half = 0.3 * reach
    found = np.array([grid_period(*orthophoto(im, data, center, x, y, half), 2 * half / 1024) for im in top])
    periods, angles = found[found[:, 2] > 0.1, :2].T  # views where the object hides most of the paper are unreliable
    keep = np.ones(len(periods), bool)
    for _ in range(3):  # a view can lock onto a multiple of the period (paper edges); keep the consistent ones
        period = np.median(periods[keep])
        keep = np.abs(periods / period - 1) < 0.2
    periods, angles = periods[keep], angles[keep]
    if not len(periods):
        sys.exit("nenhuma vista quase vertical mostra a grade")
    period = np.median(periods)
    s = spacing / period
    angle = np.median(np.mod(angles, np.pi / 2))
    gx = np.cos(angle) * x + np.sin(angle) * y
    R = np.array([gx, np.cross(z, gx), z])
    area = (np.full(2, -RULER_RADIUS_MM), np.full(2, RULER_RADIUS_MM))
    return (s, R, -s * R @ center), area, {
        "grade_mm": spacing, "vistas": len(periods), "periodo_mm_por_vista": (s * periods).tolist(),
        "desvio_periodo_pct": float(100 * np.std(periods) / period), "plano_inliers": int(inliers.sum()),
    }


def camera_axes(rec):
    images = [im for im in rec.images.values() if im.has_pose]
    return np.array([im.projection_center() for im in images]), np.array([im.cam_from_world().rotation.matrix()[2] for im in images])


def look_at_point(centers, axes):
    # Least-squares point closest to every optical axis: the object the cameras orbit.
    A = sum(np.eye(3) - np.outer(d, d) for d in axes)
    b = sum((np.eye(3) - np.outer(d, d)) @ c for c, d in zip(centers, axes))
    return np.linalg.solve(A, b)


def orbit_up(centers):
    # Cameras circle the object at a few heights, so their centers spread over a plane parallel to the table.
    return np.linalg.svd(centers - centers.mean(0))[2][-1]


def fit_plane(points, tol, up, centers, iterations=3000, max_tilt_deg=30):
    # Only planes roughly parallel to the camera orbit (a textured wall can hold more points than the table) and
    # with all cameras on one side (the top face of a box-shaped object is parallel to the table too).
    rng, best = np.random.default_rng(0), None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9 or abs(n @ up) < np.cos(np.radians(max_tilt_deg)) * np.linalg.norm(n):
            continue
        side = (centers - a) @ n > 0
        if min(side.mean(), 1 - side.mean()) > 0.02:
            continue
        inliers = np.abs((points - a) @ (n / np.linalg.norm(n))) < tol
        if best is None or inliers.sum() > best.sum():
            best = inliers
    centroid = points[best].mean(0)
    return centroid, np.linalg.svd(points[best] - centroid)[2][-1], best


def ruler_frame(rec, data):
    # Ticks of a ruler lying on the table, clicked in a few photos: {"<mm>": {"<image>": [u, v], ...}, ...}.
    ruler = measurements(data)["regua"]
    images = {im.name: im for im in rec.images.values() if im.has_pose}
    mm, ticks, errors = [], [], []
    for value, clicks in ruler.items():
        # Hand clicks are a few pixels off, unlike the subpixel ChArUco corners.
        X, e = triangulate_track([(images[n], np.asarray(uv, float) + 0.5) for n, uv in clicks.items() if n in images], 4.0)
        if X is not None:
            mm.append(float(value)), ticks.append(X), errors.extend(e)
    mm, ticks = np.array(mm), np.array(ticks)
    if len(mm) < 2:
        sys.exit(f"apenas {len(mm)} marcas da régua trianguladas")
    i, j = np.triu_indices(len(mm), 1)
    d, ref = np.linalg.norm(ticks[i] - ticks[j], axis=1), np.abs(mm[i] - mm[j])
    s = d @ ref / (d @ d)

    centers, axes = camera_axes(rec)
    centers *= s
    target = look_at_point(centers, axes)
    points = s * np.array([p.xyz for p in rec.points3D.values()])
    near = points[np.linalg.norm(points - target, axis=1) < 2 * RULER_RADIUS_MM]
    origin, z, inliers = fit_plane(near, PLANE_TOL_MM, orbit_up(centers), centers)
    if np.mean((centers - origin) @ z) < 0:
        z = -z
    x = ticks[np.argmax(mm)] - ticks[np.argmin(mm)]
    x -= (x @ z) * z
    x /= np.linalg.norm(x)
    R = np.array([x, np.cross(z, x), z])
    center = target - ((target - origin) @ z) * z
    area = (np.full(2, -RULER_RADIUS_MM), np.full(2, RULER_RADIUS_MM))
    return (s, R, -R @ center), area, {
        "marcas_usadas": len(mm),
        "residuo_regua_mm": {"rms": float(np.sqrt(((s * d - ref) ** 2).mean())), "max": float(np.abs(s * d - ref).max())},
        "marcas_ate_plano_mm": float(np.abs((s * ticks - origin) @ z).max()),
        "plano_inliers": int(inliers.sum()),
        "reprojecao_marcas_px": {"media": float(np.mean(errors)), "mediana": float(np.median(errors))},
    }


def main(data, out):
    rec = pycolmap.Reconstruction(out / "esparso")
    m = measurements(data)
    frame = (ruler_frame if "regua" in m else plane_points_frame if "pontos_plano" in m
             else grid_frame if "grade_mm" in m else board_frame)
    (s, R, t), (lo, hi), report = frame(rec, data)
    rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
    if np.median([im.projection_center()[2] for im in rec.images.values() if im.has_pose]) <= 0:
        sys.exit("câmeras abaixo da mesa: alinhamento inválido")

    (out / "metrico").mkdir(exist_ok=True)
    rec.write(out / "metrico")
    roi = rec.crop(pycolmap.AlignedBox3d(np.r_[lo - ROI_MARGIN_MM, -ROI_MARGIN_MM], np.r_[hi + ROI_MARGIN_MM, ROI_HEIGHT_MM]))
    (out / "roi").mkdir(exist_ok=True)
    roi.write(out / "roi")
    shutil.rmtree(out / "mvs", ignore_errors=True)
    pycolmap.undistort_images(out / "mvs", out / "roi", data)

    report |= {"mm_por_unidade": s, "pontos_esparsos_roi": roi.num_points3D(), "area_xy_mm": [lo.tolist(), hi.tolist()],
               "sim3": {"s": s, "R": R.tolist(), "t": t.tolist()}}
    (out / "escala.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sim3"}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
