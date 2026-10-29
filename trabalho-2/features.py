import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

MAX_SIDE, RATIO, F_THRESHOLD_PX = 1600, 0.75, 1.0
GAPS = (1, 3)


def detectors():
    return {"SIFT": cv2.SIFT_create(8000), "ORB": cv2.ORB_create(8000), "AKAZE": cv2.AKAZE_create()}


def matcher(name):
    if name == "SIFT":
        return cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    return cv2.BFMatcher(cv2.NORM_HAMMING)


def ratio_matches(m, d1, d2, ratio):
    return {p[0].queryIdx: p[0].trainIdx for p in m.knnMatch(d1, d2, k=2)
            if len(p) == 2 and p[0].distance < ratio * p[1].distance}


def mutual_matches(m, d1, d2, ratio=RATIO):
    forward, backward = ratio_matches(m, d1, d2, ratio), ratio_matches(m, d2, d1, ratio)
    return np.array([(i, j) for i, j in forward.items() if backward.get(j) == i]).reshape(-1, 2)


def load(path, max_side=MAX_SIDE):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def rotation_angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def sampson(F, p1, p2):
    x1, x2 = np.c_[p1, np.ones(len(p1))], np.c_[p2, np.ones(len(p2))]
    Fx1, Ftx2 = x1 @ F.T, x2 @ F
    return (np.sum(x2 * Fx1, 1) ** 2) / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2)


def two_view(name, det, im1, im2, img1, img2, s, ratio=RATIO, threshold=F_THRESHOLD_PX):
    t0 = time.perf_counter()
    k1, d1 = det.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = det.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    detect_ms = (time.perf_counter() - t0) * 500
    pairs = mutual_matches(matcher(name), d1, d2, ratio)
    p1 = np.float64([k1[i].pt for i in pairs[:, 0]])
    p2 = np.float64([k2[j].pt for j in pairs[:, 1]])
    F, mask_f = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, threshold, 0.999)
    inl = mask_f.ravel() > 0

    cam = im1.camera
    n1, n2 = cam.cam_from_img((p1 + 0.5) / s), cam.cam_from_img((p2 + 0.5) / s)
    E, mask_e = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.999, threshold / (cam.mean_focal_length() * s))
    _, R, t, mask_pose = cv2.recoverPose(E, n1, n2, np.eye(3), mask=mask_e.copy())
    good = mask_pose.ravel() > 0
    T1, T2 = im1.cam_from_world().matrix(), im2.cam_from_world().matrix()
    R_ref = T2[:, :3] @ T1[:, :3].T
    t_ref = T2[:, 3] - R_ref @ T1[:, 3]
    t_err = np.degrees(np.arccos(np.clip(t.ravel() @ t_ref / np.linalg.norm(t_ref), -1, 1)))

    Xh = cv2.triangulatePoints(np.eye(3, 4), np.c_[R, t], n1[good].T, n2[good].T)
    X = (Xh[:3] / Xh[3]).T
    front = (X[:, 2] > 0) & ((X @ R.T + t.ravel())[:, 2] > 0)
    proj = X @ R.T + t.ravel()
    reproj = np.linalg.norm(proj[:, :2] / proj[:, 2:] - n2[good], axis=1) * cam.mean_focal_length()
    stats = {
        "keypoints": [len(k1), len(k2)], "deteccao_ms_por_imagem": detect_ms, "matches": len(pairs),
        "inliers_F": int(inl.sum()), "inliers_F_pct": 100 * inl.mean(),
        "sampson_mediana_px": float(np.median(np.sqrt(sampson(F, p1[inl], p2[inl])))),
        "inliers_E": int(good.sum()), "erro_R_graus": float(rotation_angle(R @ R_ref.T)), "erro_t_graus": float(t_err),
        "triangulados_frente": int(front.sum()), "reprojecao_tri_px": float(np.median(reproj)),
    }
    return stats, dict(k1=k1, k2=k2, pairs=pairs, inl=inl, F=F, p1=p1, p2=p2, X=X[front], c=img1[p1[good][front][:, 1].astype(int), p1[good][front][:, 0].astype(int)])


def draw_keypoints(img, results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5))
    for ax, (name, (stats, r)) in zip(np.atleast_1d(axes), results.items()):
        ax.imshow(cv2.cvtColor(cv2.drawKeypoints(img, r["k1"], None, (0, 255, 0)), cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}: {stats['keypoints'][0]} keypoints, {stats['deteccao_ms_por_imagem']:.0f} ms")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_inliers(img1, img2, r, path, n=300):
    idx = np.random.default_rng(0).permutation(len(r["pairs"]))[:n]
    matches = [cv2.DMatch(int(i), int(j), 0) for i, j in r["pairs"][idx]]
    vis = cv2.drawMatches(img1, r["k1"], img2, r["k2"], matches, None, matchesMask=r["inl"][idx].astype(int).tolist(),
                          matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    outl = [m for m, ok in zip(matches, r["inl"][idx]) if not ok]
    vis = cv2.drawMatches(img1, r["k1"], img2, r["k2"], outl, vis, matchColor=(0, 0, 255),
                          flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(path), vis)


def draw_epipolar(img1, img2, r, path, n=15):
    a, b = img1.copy(), img2.copy()
    inl = np.flatnonzero(r["inl"])
    rng = np.random.default_rng(1)
    for k in rng.choice(inl, min(n, len(inl)), replace=False):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        for img, p, q, F in ((b, r["p1"][k], r["p2"][k], r["F"]), (a, r["p2"][k], r["p1"][k], r["F"].T)):
            l = F @ np.r_[p, 1]
            x0, x1 = 0, img.shape[1]
            cv2.line(img, (x0, int(-l[2] / l[1])), (x1, int(-(l[2] + l[0] * x1) / l[1])), color, 2)
            cv2.circle(img, tuple(int(v) for v in q), 6, color, -1)
    cv2.imwrite(str(path), np.hstack([a, b]))


def draw_triangulation(r, path):
    X, c = r["X"], r["c"][:, ::-1] / 255
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X[:, 0], X[:, 2], -X[:, 1], c=c, s=1)
    ax.set_xlabel("x")
    ax.set_ylabel("z (profundidade)")
    ax.set_zlabel("-y")
    ax.set_box_aspect(np.ptp(X[:, [0, 2, 1]], 0))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(data, out, names=None):
    rec = pycolmap.Reconstruction(out / "metrico")
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    ring = sorted(n for n in by_name if n.startswith("h2/")) or sorted(by_name)
    pairs = [names] if names else [(ring[0], ring[g]) for g in GAPS]
    figures = out / "figuras"
    figures.mkdir(exist_ok=True)
    table = {}
    for n1, n2 in pairs:
        (img1, s), (img2, _) = load(data / n1), load(data / n2)
        results = {}
        for name, det in detectors().items():
            try:
                results[name] = two_view(name, det, by_name[n1], by_name[n2], img1, img2, s)
            except (cv2.error, TypeError, ValueError, AttributeError) as e:  # too few matches for F, E or triangulation
                print(f"{n1} x {n2}: {name} falhou ({e})")
        if "SIFT" not in results:
            continue
        tag = f"{Path(n1).stem}_{Path(n2).stem}"
        draw_keypoints(img1, results, figures / f"keypoints_{tag}.png")
        for name, (_, r) in results.items():
            draw_inliers(img1, img2, r, figures / f"inliers_{name}_{tag}.jpg")
        draw_epipolar(img1, img2, results["SIFT"][1], figures / f"epipolares_SIFT_{tag}.jpg")
        draw_triangulation(results["SIFT"][1], figures / f"triangulacao_SIFT_{tag}.png")
        table[f"{n1} x {n2}"] = {name: stats for name, (stats, _) in results.items()}
    (out / "features.json").write_text(json.dumps(table, indent=2))
    for pair, rows in table.items():
        print(pair)
        for name, st in rows.items():
            print(f"  {name:5s} kp {st['keypoints'][0]:5d} {st['deteccao_ms_por_imagem']:6.0f} ms | matches {st['matches']:5d} "
                  f"F {st['inliers_F']:5d} ({st['inliers_F_pct']:4.1f}%) sampson {st['sampson_mediana_px']:.2f} | "
                  f"E {st['inliers_E']:5d} dR {st['erro_R_graus']:.2f} dt {st['erro_t_graus']:.2f} | tri {st['triangulados_frente']} {st['reprojecao_tri_px']:.2f} px")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), tuple(sys.argv[3:5]) if len(sys.argv) > 4 else None)
