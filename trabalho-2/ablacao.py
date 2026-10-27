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

CONFIGS = {
    "3 alturas, todas": {"h1": 1, "h2": 1, "h3": 1, "topo": 1},
    "3 alturas, 1/2": {"h1": 2, "h2": 2, "h3": 2, "topo": 1},
    "3 alturas, 1/3": {"h1": 3, "h2": 3, "h3": 3, "topo": 1},
    "3 alturas, 1/4": {"h1": 4, "h2": 4, "h3": 4, "topo": 1},
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
    for folder, step in steps.items():
        files = sorted(p for p in (data / folder).glob("*") if p.suffix.lower() in EXTENSIONS)
        (dst / folder).mkdir(parents=True)
        for p in files[::step]:
            os.symlink(p.resolve(), dst / folder / p.name)
            names.append(f"{folder}/{p.name}")
    if (data / "medidas.json").exists():
        os.symlink((data / "medidas.json").resolve(), dst / "medidas.json")
    return names


def distances(src, dst_mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(dst_mesh))
    return scene.compute_distance(o3d.core.Tensor(src.astype(np.float32))).numpy()


def samples(mesh):
    o3d.utility.random.seed(0)
    return np.asarray(mesh.sample_points_uniformly(max(int(mesh.get_surface_area()), 1000)).points)


def main(data, out):
    results = {}
    for name, steps in CONFIGS.items():
        subset, result = out / slug(name) / "dados", out / slug(name) / "saida"
        names = make_subset(data, steps, subset)
        manifest = {"fotos": names, "quadrado_mm": escala.board_points(subset)[1]}
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
        mesh = o3d.io.read_triangle_mesh(str(out / slug(name) / "saida" / "objeto.ply"))
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
        ax.set_xticks(x, [f"{n}\n({results[n]['fotos']} fotos)" for n in names], fontsize=8)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).parent / "saida" / f"{Path(sys.argv[1]).name}_ablacao")
