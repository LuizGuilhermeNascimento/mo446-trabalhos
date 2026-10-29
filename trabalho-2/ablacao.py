import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import escala
from sfm import EXTENSIONS

# "extra" is every folder besides the rings h1-h3 (topo for the mill, detalhe for the duck); a step of N keeps every Nth photo.
CONFIGS = {
    "3 alturas, todas": {"h1": 1, "h2": 1, "h3": 1, "extra": 1},
    "3 alturas, 1/2": {"h1": 2, "h2": 2, "h3": 2, "extra": 1},
    "3 alturas, 1/3": {"h1": 3, "h2": 3, "h3": 3, "extra": 1},
    "3 alturas, 1/4": {"h1": 4, "h2": 4, "h3": 4, "extra": 1},
    "3 alturas, sem extra": {"h1": 1, "h2": 1, "h3": 1},
    "h1+h2": {"h1": 1, "h2": 1},
    "h2+h3": {"h2": 1, "h3": 1},
    "só h2": {"h2": 1},
}
TAUS_MM = (1.0, 2.0)


def slug(name):
    return "".join(c if c.isascii() and c.isalnum() else "_" for c in name)


def make_subset(data, steps, dst):
    shutil.rmtree(dst, ignore_errors=True)
    names = []
    # Photos with hand-clicked scale marks stay in every subset, so the ablation varies the photos, not the reference.
    m = escala.measurements(data)
    clicked = {n for marks in (m.get("regua") or m.get("pontos_plano") or {}).values() for n in marks}
    extra = sorted(p.name for p in data.iterdir() if p.is_dir() and p.name not in ("h1", "h2", "h3", "calib"))
    for folder, step in [(f, s) for k, s in steps.items() for f in (extra if k == "extra" else [k])]:
        files = sorted(p for p in (data / folder).glob("*") if p.suffix.lower() in EXTENSIONS)
        (dst / folder).mkdir(parents=True)
        for p in [f for k, f in enumerate(files) if k % step == 0 or f"{folder}/{f.name}" in clicked]:
            os.symlink(p.resolve(), dst / folder / p.name)
            names.append(f"{folder}/{p.name}")
    for name in sorted(clicked - set(names)):
        (dst / name).parent.mkdir(parents=True, exist_ok=True)
        os.symlink((data / name).resolve(), dst / name)
        names.append(name)
    if (data / "medidas.json").exists():
        os.symlink((data / "medidas.json").resolve(), dst / "medidas.json")
    return names


def distances(src, dst_mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(dst_mesh))
    return scene.compute_distance(o3d.core.Tensor(src.astype(np.float32))).numpy()


def align(mesh, ref_pts, threshold_mm=3.0):
    # Each subset gets its own metric frame (and, with a ruler, its own origin), so a millimetre-level rigid offset
    # would count as missing surface; compare shapes after a rigid ICP, as benchmarks like Tanks and Temples do.
    src, dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(samples(mesh))), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_pts))
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
    fit = o3d.pipelines.registration.registration_icp(src, dst, threshold_mm, np.eye(4),
                                                      o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return mesh.transform(fit.transformation), float(np.linalg.norm(fit.transformation[:3, 3]))


def samples(mesh):
    o3d.utility.random.seed(0)
    return np.asarray(mesh.sample_points_uniformly(max(int(mesh.get_surface_area()), 1000)).points)


def main(data, out):
    results = {}
    for name, steps in CONFIGS.items():
        subset, result = out / slug(name) / "dados", out / slug(name) / "saida"
        names = make_subset(data, steps, subset)
        manifest = {"fotos": names, "escala": {k: v for k, v in escala.measurements(subset).items() if k in ("quadrado_mm", "regua")}}
        stamp = result / "manifest.json"
        t0 = time.time()
        if stamp.exists() and json.loads(stamp.read_text()) == manifest and (result / "metricas.json").exists():
            # Same photos and board: reuse the geometry, but redo the metrics in case the measurements changed.
            ok = subprocess.run([sys.executable, "avaliar.py", subset, result], cwd=Path(__file__).parent).returncode == 0
        else:
            shutil.rmtree(result, ignore_errors=True)
            ok = subprocess.run(["./run.sh", subset, result], env=os.environ | {"RAPIDO": "1", "RES": "2"},
                                cwd=Path(__file__).parent).returncode == 0
            if ok:
                stamp.write_text(json.dumps(manifest))
        results[name] = {"fotos": len(names), "ok": ok and (result / "metricas.json").exists(), "minutos": (time.time() - t0) / 60}
        if results[name]["ok"]:
            m = json.loads((result / "metricas.json").read_text())
            results[name] |= {"registradas": m["imagens_registradas"], "reprojecao_px": m["reprojecao_px"]["mediana"],
                              "dimensoes": {k: v.get("erro_abs") for k, v in m["dimensoes"].items()},
                              "estimado": {k: v["estimado"] for k, v in m["dimensoes"].items()}}
        print(name, json.dumps(results[name], ensure_ascii=False), flush=True)

    ref = o3d.io.read_triangle_mesh(str(out / slug(next(iter(CONFIGS))) / "saida" / "objeto.ply"))
    ref_pts = samples(ref)
    for name, r in results.items():
        if not r["ok"]:
            continue
        mesh, r["icp_translacao_mm"] = align(o3d.io.read_triangle_mesh(str(out / slug(name) / "saida" / "objeto.ply")), ref_pts)
        to_ref, from_ref = distances(samples(mesh), ref), distances(ref_pts, mesh)
        for tau in TAUS_MM:
            r[f"acuracia_{tau:g}mm_pct"] = float(100 * (to_ref < tau).mean())
            r[f"completude_{tau:g}mm_pct"] = float(100 * (from_ref < tau).mean())
        r["chamfer_mm"] = float((to_ref.mean() + from_ref.mean()) / 2)
    (out / "ablacao.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    plot(results, out / "ablacao.png")


def plot(results, path):
    names = [n for n, r in results.items() if r["ok"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(names))
    keys = [k for k in ("comprimento_mm", "largura_mm", "altura_mm") if all(results[n]["dimensoes"].get(k) is not None for n in names)]
    for i, k in enumerate(keys):
        axes[0].bar(x + (i - 1) * 0.25, [results[n]["dimensoes"][k] for n in names], 0.25, label=k.split("_")[0])
    axes[0].set_ylabel("erro absoluto (mm)")
    axes[1].bar(x - 0.2, [results[n]["completude_1mm_pct"] for n in names], 0.4, label="completude (τ = 1 mm)")
    axes[1].bar(x + 0.2, [results[n]["acuracia_1mm_pct"] for n in names], 0.4, label="acurácia (τ = 1 mm)")
    axes[1].set_ylabel("% em relação à reconstrução completa")
    for ax in axes:
        ax.set_xticks(x, [f"{n} ({results[n]['fotos']})" for n in names], fontsize=8, rotation=30, ha="right")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).parent / "saida" / f"{Path(sys.argv[1]).name}_ablacao")
