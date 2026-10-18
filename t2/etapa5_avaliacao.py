"""Métricas de qualidade da reconstrução produzida na Etapa 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import open3d as o3d
import numpy as np
import cv2





def _ler_pontos_3d(caminho: Path) -> dict[int, Any]:
    pontos = {}
    for linha in caminho.read_text(encoding="utf-8", errors="replace").splitlines():
        campos = linha.split()
        if len(campos) >= 8 and not campos[0].startswith("#"):
            pontos[int(campos[0])] = np.asarray(campos[1:4], dtype=np.float64)
    return pontos


def _calcular_erros_reprojecao(text_model: Path) -> dict[str, float | None]:
    """Calcula os resíduos de reprojeção de todas as observações 3D registradas."""
    cameras = {}
    for line in (text_model / "cameras.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0].startswith("#"):
            continue
        camera_id, model = int(fields[0]), fields[1]
        width, height = map(int, fields[2:4])
        params = np.asarray(fields[4:], dtype=np.float64)
        if model == "OPENCV" and len(params) >= 8:
            fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist = np.array([k1, k2, p1, p2], dtype=np.float64)
        elif model == "PINHOLE" and len(params) >= 4:
            fx, fy, cx, cy = params[:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist = np.zeros(4)
        else:
            continue
        cameras[camera_id] = (K, dist, width, height)
    points3d = _ler_pontos_3d(text_model / "points3D.txt")
    residuals = []
    rows = (text_model / "images.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    idx = 0
    while idx < len(rows):
        header = rows[idx].strip(); idx += 1
        if not header or header.startswith("#"):
            continue
        fields = header.split()
        if len(fields) < 10:
            continue
        camera_id = int(fields[8])
        observations = rows[idx].split() if idx < len(rows) else []; idx += 1
        if camera_id not in cameras or not observations:
            continue
        obs = np.asarray(observations, dtype=np.float64).reshape(-1, 3)
        ids = obs[:, 2].astype(np.int64)
        keep = np.array([point_id in points3d for point_id in ids])
        if not np.any(keep):
            continue
        q = np.asarray(fields[1:5], dtype=np.float64)  # componentes escalar, x, y e z
        qw, qx, qy, qz = q
        rotation = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = np.asarray(fields[5:8], dtype=np.float64).reshape(3, 1)
        object_points = np.asarray([points3d[int(point_id)] for point_id in ids[keep]], dtype=np.float64)
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, cameras[camera_id][0], cameras[camera_id][1])
        residuals.extend(np.linalg.norm(projected.reshape(-1, 2) - obs[keep, :2], axis=1).tolist())
    return {
        "mean_px": float(np.mean(residuals)) if residuals else None,
        "median_px": float(np.median(residuals)) if residuals else None,
    }


def avaliar_etapa5(
    relatorio_etapa4: dict[str, Any],
    *,
    output_dir: str | Path,
    expected_dimensions_mm: dict[str, float] | None = None,
    expected_volume_cm3: float | None = None,
) -> dict[str, Any]:
    """Calcula erros e densidades a partir dos arquivos métricos da Etapa 4."""
    output = Path(output_dir)
    frame_tag = relatorio_etapa4["frame_tag"]
    sparse_text = Path(relatorio_etapa4["sparse_text_model"])
    mesh = o3d.io.read_triangle_mesh(relatorio_etapa4["mesh_metric"], enable_post_processing=True)
    if mesh.is_empty():
        raise ValueError("A malha métrica da Etapa 4 está vazia ou não pode ser lida.")

    dimension_names = ("maior", "intermediaria", "menor")
    dimensions_mm = np.sort(np.asarray(mesh.get_axis_aligned_bounding_box().get_extent()) * 1000.0)[::-1]
    dimension_errors = {}
    for name, measured in zip(dimension_names, dimensions_mm):
        expected = (expected_dimensions_mm or {}).get(name)
        absolute = abs(float(measured) - float(expected)) if expected is not None else None
        dimension_errors[name] = {
            "medida_mm": float(measured),
            "referencia_mm": float(expected) if expected is not None else None,
            "erro_absoluto_mm": absolute,
            "erro_relativo_percentual": absolute / float(expected) * 100.0 if absolute is not None and expected else None,
        }

    is_watertight = bool(mesh.is_watertight())
    volume_cm3 = float(mesh.get_volume()) * 1e6 if is_watertight else None
    volume_absolute_error = abs(volume_cm3 - expected_volume_cm3) if volume_cm3 is not None and expected_volume_cm3 else None
    surface_area_m2 = float(mesh.get_surface_area())
    reprojection = _calcular_erros_reprojecao(sparse_text)
    sparse_points = len(_ler_pontos_3d(sparse_text / "points3D.txt"))

    def point_count(key: str) -> int:
        cloud = o3d.io.read_point_cloud(relatorio_etapa4[key])
        if cloud.is_empty():
            raise ValueError(f"Nuvem sem pontos: {relatorio_etapa4[key]}")
        return len(cloud.points)

    dense_geometric = point_count("dense_geometric_cloud")
    dense_photometric = point_count("dense_photometric_cloud")
    report = {
        "etapa": 5,
        "quadros": frame_tag,
        "erro_reprojecao_media_px": reprojection["mean_px"],
        "erro_reprojecao_mediana_px": reprojection["median_px"],
        "pontos_esparsos": sparse_points,
        "pontos_densos_geometricos": dense_geometric,
        "pontos_densos_fotometricos": dense_photometric,
        "area_superficial_m2": surface_area_m2,
        "densidade_geometrica_pontos_por_m2": dense_geometric / surface_area_m2 if surface_area_m2 > 0 else None,
        "densidade_fotometrica_pontos_por_m2": dense_photometric / surface_area_m2 if surface_area_m2 > 0 else None,
        "dimensoes_e_erros": dimension_errors,
        "volume_malha_cm3": volume_cm3,
        "volume_referencia_cm3": float(expected_volume_cm3) if expected_volume_cm3 is not None else None,
        "erro_absoluto_volume_cm3": volume_absolute_error,
        "erro_relativo_volume_percentual": volume_absolute_error / expected_volume_cm3 * 100.0 if volume_absolute_error is not None and expected_volume_cm3 else None,
        "malha_fechada": is_watertight,
        "comparacao_foto_render": relatorio_etapa4["side_by_side"],
        "malha_texturizada": relatorio_etapa4["textured_metric_mesh"],
    }
    (output / f"relatorio_etapa5_{frame_tag}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
