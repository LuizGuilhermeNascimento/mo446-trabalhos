import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pycolmap

import ablacao
import avaliar
import escala
import features

RATIOS = (0.6, 0.7, 0.75, 0.8, 0.9)
THRESHOLDS_PX = (0.25, 0.5, 1.0, 2.0, 4.0)
SIDES = (800, 1600, 3200)
CUTS_MM = (1.0, 2.0, 3.0, 5.0)
SUBSETS, DRAWS = (4, 8, 16), 50
# Pipeline reruns: each changes one knob from the RAPIDO baseline (RES=2, RADIAL, SIFT at 3200 px).
RUNS = {
    "base": {},
    "MVS res 1": {"RES": "1"},
    "MVS res 3": {"RES": "3"},
    "SIMPLE_RADIAL": {"CAMERA": "SIMPLE_RADIAL"},
    "OPENCV": {"CAMERA": "OPENCV"},
    "SIFT 1600 px": {"SIFT_MAX": "1600"},
}


def two_view_sweep(data, out, pairs_per_gap=6):
    rec = pycolmap.Reconstruction(out / "metrico")
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    ring = sorted(n for n in by_name if n.startswith("h2/"))
    pairs = [(gap, ring[i], ring[i + gap]) for gap in (1, 2)
             for i in np.linspace(0, len(ring) - gap - 1, pairs_per_gap).astype(int)]
    dets = features.detectors()
    rows = []
    for side in SIDES:
        base = side == features.MAX_SIDE
        grid = [(r, features.F_THRESHOLD_PX) for r in (RATIOS if base else (features.RATIO,))]
        grid += [(features.RATIO, t) for t in (THRESHOLDS_PX if base else ()) if t != features.F_THRESHOLD_PX]
        for gap, n1, n2 in pairs:
            (img1, s), (img2, _) = features.load(data / n1, side), features.load(data / n2, side)
            for name, det in dets.items():
                for ratio, threshold in grid:
                    try:
                        st, _ = features.two_view(name, det, by_name[n1], by_name[n2], img1, img2, s, ratio, threshold)
                    except (cv2.error, AttributeError, ValueError):  # too few matches for a single F
                        continue
                    rows.append({"par": f"{n1} x {n2}", "gap": gap, "lado_px": side, "detector": name,
                                 "ratio": ratio, "limiar_px": threshold} | st)
    return rows


def scale_sweep(data, out):
    rec = pycolmap.Reconstruction(out / "esparso")
    m = escala.measurements(data)
    if "regua" in m:
        images = {im.name: im for im in rec.images.values() if im.has_pose}
        mm, ticks = [], []
        for value, clicks in m["regua"].items():
            X, _ = escala.triangulate_track([(images[n], np.asarray(uv, float) + 0.5) for n, uv in clicks.items()], 4.0)
            if X is not None:
                mm.append(float(value)), ticks.append(X)
        mm, ticks = np.array(mm), np.array(ticks)

        def scale(sel):
            i, j = np.triu_indices(len(sel), 1)
            d, ref = np.linalg.norm(ticks[sel][i] - ticks[sel][j], axis=1), np.abs(mm[sel][i] - mm[sel][j])
            return d @ ref / (d @ d)
        full = scale(np.arange(len(mm)))
        return {"tipo": "régua", "s_completo": full, "deixa_um_fora": {
            f"sem {mm[k]:g} mm": 100 * (scale(np.delete(np.arange(len(mm)), k)) / full - 1) for k in range(len(mm))}}
    board, _ = escala.board_points(data)
    model, target = [], []
    for i, track in escala.corner_tracks(rec, data).items():
        X, _ = escala.triangulate_track(track) if len(track) >= 3 else (None, None)
        if X is not None:
            model.append(X), target.append(board[i])
    model, target = np.array(model), np.array(target)
    full = escala.umeyama(model, target)[0]
    rng = np.random.default_rng(0)
    rows = {}
    for k in SUBSETS:
        s = [escala.umeyama(model[idx], target[idx])[0] for idx in (rng.choice(len(model), k, replace=False) for _ in range(DRAWS))]
        rows[f"{k} cantos"] = {"desvio_pct": float(100 * np.std(s) / full), "max_pct": float(100 * np.max(np.abs(np.array(s) / full - 1)))}
    return {"tipo": "tabuleiro", "cantos": len(model), "s_completo": full, "subconjuntos": rows}


def cut_sweep(data, out):
    metrics = json.loads((out / "metricas.json").read_text())
    z0 = metrics["tabuleiro_z_mm"]["mediana"]
    lo, hi = np.array(json.loads((out / "escala.json").read_text())["area_xy_mm"]) + [[-avaliar.MARGIN_MM], [avaliar.MARGIN_MM]]
    rows = {}
    for cut in CUTS_MM:
        avaliar.CUT_MM = cut
        obj = avaliar.crop_object(o3d.io.read_triangle_mesh(str(out / "mvs/scene_mesh.ply")), lo, hi, z0)
        V = np.asarray(obj.vertices)
        _, (a, b), _ = cv2.minAreaRect(V[:, :2].astype(np.float32))
        rows[f"{cut:g} mm"] = {"comprimento_mm": float(max(a, b)), "largura_mm": float(min(a, b)),
                               "altura_mm": float(V[:, 2].max() - z0), "volume_cm3": avaliar.volume_cm3(obj, z0)}
    return rows


def pipeline_runs(data, out, dst):
    ref = o3d.io.read_triangle_mesh(str(out / "objeto.ply"))
    ref_pts = ablacao.samples(ref)
    rows = {}
    for name, env in RUNS.items():
        result = dst / ablacao.slug(name)
        stamp, manifest = result / "manifest.json", {"env": env, "escala": {k: v for k, v in escala.measurements(data).items() if k in ("quadrado_mm", "regua")}}
        t0 = time.time()
        cached = json.loads(stamp.read_text()) if stamp.exists() else {}
        if not ({k: v for k, v in cached.items() if k != "minutos"} == manifest and (result / "metricas.json").exists()):
            shutil.rmtree(result, ignore_errors=True)
            run_env = os.environ | {"RAPIDO": "1", "RES": "2"} | env
            if subprocess.run(["./run.sh", data, result], env=run_env, cwd=Path(__file__).parent).returncode != 0:
                rows[name] = {"ok": False}
                continue
            stamp.write_text(json.dumps(manifest | {"minutos": (time.time() - t0) / 60}))
        m = json.loads((result / "metricas.json").read_text())
        mesh, shift = ablacao.align(o3d.io.read_triangle_mesh(str(result / "objeto.ply")), ref_pts)
        to_ref, from_ref = ablacao.distances(ablacao.samples(mesh), ref), ablacao.distances(ref_pts, mesh)
        rows[name] = {"ok": True, "minutos": json.loads(stamp.read_text()).get("minutos"),
                      "registradas": m["imagens_registradas"], "reprojecao_px": m["reprojecao_px"]["mediana"],
                      "pontos_densos_objeto": m["pontos_densos_objeto"],
                      "dimensoes": {k: v["estimado"] for k, v in m["dimensoes"].items()},
                      "erro_mm": {k: v.get("erro_abs") for k, v in m["dimensoes"].items()},
                      "acuracia_1mm_pct": float(100 * (to_ref < 1).mean()), "completude_1mm_pct": float(100 * (from_ref < 1).mean()),
                      "icp_translacao_mm": shift}
        print(name, json.dumps(rows[name], ensure_ascii=False), flush=True)
    return rows


def median_curve(rows, x, y, **fixed):
    sel = [r for r in rows if all(r[k] == v for k, v in fixed.items())]
    xs = sorted({r[x] for r in sel})
    return xs, [float(np.median([r[y] for r in sel if r[x] == v])) for v in xs]


def plot(report, path):
    rows = report["dois_pontos"]
    base = {"lado_px": features.MAX_SIDE}
    panels = [
        ("ratio", "inliers_F", "limiar do ratio test", "inliers de F", base | {"limiar_px": features.F_THRESHOLD_PX}),
        ("ratio", "inliers_F_pct", "limiar do ratio test", "inliers / correspondências (%)", base | {"limiar_px": features.F_THRESHOLD_PX}),
        ("ratio", "erro_R_graus", "limiar do ratio test", "erro de rotação vs. SfM (graus)", base | {"limiar_px": features.F_THRESHOLD_PX}),
        ("limiar_px", "inliers_F_pct", "limiar do RANSAC (px)", "inliers / correspondências (%)", base | {"ratio": features.RATIO}),
        ("limiar_px", "erro_t_graus", "limiar do RANSAC (px)", "erro de direção de t vs. SfM (graus)", base | {"ratio": features.RATIO}),
        ("lado_px", "inliers_F", "maior lado da imagem (px)", "inliers de F", {"ratio": features.RATIO, "limiar_px": features.F_THRESHOLD_PX}),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (x, y, xlabel, ylabel, fixed) in zip(axes.flat, panels):
        for name in features.detectors():
            for gap, style in ((1, "o-"), (2, "s--")):
                xs, ys = median_curve(rows, x, y, detector=name, gap=gap, **fixed)
                ax.plot(xs, ys, style, color=f"C{list(features.detectors()).index(name)}", label=f"{name}, {gap} passo{'s' * (gap > 1)}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if x == "limiar_px":
            ax.set_xscale("log", base=2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Mediana sobre pares do anel h2 (linha cheia: vizinhos; tracejada: 2 passos)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path / "hiperparametros_2vistas.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    cuts = report["corte"]
    x = [float(k.split()[0]) for k in cuts]
    for key in ("comprimento_mm", "largura_mm", "altura_mm"):
        axes[0].plot(x, [v[key] for v in cuts.values()], "o-", label=key.split("_")[0])
    axes[0].set_xlabel("altura do corte acima da mesa (mm)")
    axes[0].set_ylabel("dimensão estimada (mm)")
    axes[0].legend(fontsize=8)
    sc = report["escala"]
    if sc["tipo"] == "tabuleiro":
        k = [int(n.split()[0]) for n in sc["subconjuntos"]]
        axes[1].plot(k, [v["desvio_pct"] for v in sc["subconjuntos"].values()], "o-", label="desvio-padrão")
        axes[1].plot(k, [v["max_pct"] for v in sc["subconjuntos"].values()], "s--", label="maior desvio")
        axes[1].set_xlabel(f"cantos do ChArUco usados (de {sc['cantos']})")
        axes[1].legend(fontsize=8)
    else:
        axes[1].bar(range(len(sc["deixa_um_fora"])), list(sc["deixa_um_fora"].values()))
        axes[1].set_xticks(range(len(sc["deixa_um_fora"])), list(sc["deixa_um_fora"]), fontsize=8)
        axes[1].set_xlabel("marca da régua retirada")
    axes[1].set_ylabel("variação da escala (%)")
    fig.tight_layout()
    fig.savefig(path / "hiperparametros_escala_corte.png", dpi=130)
    plt.close(fig)


def main(data, out, heavy):
    dst = out.parent / f"{out.name}_parametros"
    dst.mkdir(exist_ok=True)
    path = dst / "parametros.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report |= {"dois_pontos": two_view_sweep(data, out), "escala": scale_sweep(data, out), "corte": cut_sweep(data, out)}
    if heavy:
        report["pipeline"] = pipeline_runs(data, out, dst)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    plot(report, dst)
    print(json.dumps({k: v for k, v in report.items() if k != "dois_pontos"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), "--pipeline" in sys.argv)
