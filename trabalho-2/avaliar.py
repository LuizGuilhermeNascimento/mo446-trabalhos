import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap

import sfm

CUT_MM, MARGIN_MM, NEAR_MM = 2.0, 10.0, 1.0


def reprojection_errors(rec):
    errors = []
    for image in rec.images.values():
        if image.has_pose:
            xy, X = map(np.array, zip(*[(p.xy, rec.points3D[p.point3D_id].xyz) for p in image.points2D if p.has_point3D()]))
            errors.append(np.linalg.norm(image.camera.img_from_cam(image.cam_from_world() * X) - xy, axis=1))
    return np.concatenate(errors)


def crop_object(mesh, lo, hi, z0):
    V = np.asarray(mesh.vertices)
    mesh.remove_vertices_by_mask(~(np.all(V[:, :2] > lo, 1) & np.all(V[:, :2] < hi, 1) & (V[:, 2] > z0 + CUT_MM)))
    labels, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(labels) != np.argmax(counts))
    mesh.remove_unreferenced_vertices()
    return mesh


def volume_cm3(mesh, z0):
    # Flux of F = (0, 0, z - z0) through the surface; the base resting on the table (z = z0) adds nothing,
    # assuming the walls go straight down from the 2 mm cut to the table.
    V, T = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    a, b, c = (V[T[:, k]] for k in range(3))
    v = abs(((a[:, 2] + b[:, 2] + c[:, 2]) / 3 - z0) @ (0.5 * np.cross(b - a, c - a)[:, 2])) / 1000
    return float(v) if np.isfinite(v) else None


def compare(estimated, path):
    measured = json.loads(path.read_text()) if path.exists() else {}
    result = {}
    for key, value in estimated.items():
        result[key] = {"estimado": value}
        if any(measured.get(key) or []):
            ref = float(np.mean(measured[key]))
            result[key] |= {"medido": ref, "desvio_medidas": float(np.std(measured[key])),
                            "erro_abs": abs(value - ref), "erro_rel_pct": 100 * abs(value - ref) / ref}
    return result


def main(data, out):
    rec = pycolmap.Reconstruction(out / "metrico")
    errors = reprojection_errors(rec)
    dense = np.asarray(o3d.io.read_point_cloud(str(out / "mvs/scene_dense.ply")).points)
    lo, hi = np.array(json.loads((out / "escala.json").read_text())["area_xy_mm"]) + [[-MARGIN_MM], [MARGIN_MM]]
    on_board = np.all(dense[:, :2] > lo, 1) & np.all(dense[:, :2] < hi, 1)
    plane = dense[on_board & (np.abs(dense[:, 2]) < 3), 2]
    z0 = float(np.median(plane))

    obj = crop_object(o3d.io.read_triangle_mesh(str(out / "mvs/scene_mesh.ply")), lo, hi, z0)
    o3d.io.write_triangle_mesh(str(out / "objeto.ply"), obj)
    V = np.asarray(obj.vertices)
    _, (a, b), _ = cv2.minAreaRect(V[:, :2].astype(np.float32))
    estimated = {"comprimento_mm": max(a, b), "largura_mm": min(a, b), "altura_mm": V[:, 2].max() - z0}
    if (volume := volume_cm3(obj, z0)) is not None:
        estimated["volume_cm3"] = volume

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(obj))
    above = dense[on_board & (dense[:, 2] > z0 + CUT_MM)].astype(np.float32)
    on_object = int((scene.compute_distance(o3d.core.Tensor(above)).numpy() < NEAR_MM).sum())
    area_m2 = obj.get_surface_area() / 1e6

    report = {
        "imagens": len(sfm.image_names(data)),
        "imagens_registradas": rec.num_reg_images(),
        "reprojecao_px": {"media": float(errors.mean()), "mediana": float(np.median(errors))},
        "pontos_esparsos": rec.num_points3D(),
        "pontos_densos": len(dense),
        "pontos_densos_objeto": on_object,
        "area_objeto_m2": area_m2,
        "densidade_pontos_m2": on_object / area_m2,
        "tabuleiro_z_mm": {"mediana": z0, "desvio": float(plane.std())},
        "escala": {k: v for k, v in json.loads((out / "escala.json").read_text()).items() if k != "sim3"},
        "dimensoes": compare(estimated, data / "medidas.json"),
    }
    (out / "metricas.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
