from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from common import (IMAGE_SUPTITLE, IMAGE_TEXT, IMAGE_TITLE, INK_SECONDARY, SERIES_COLORS, log, pair_name,
                    print_table, save_figure, write_csv)
from step4_homography_connection.homography import PairHomography, fit_homography, plot_ransac, reprojection_errors

STAGE = Path(__file__).resolve().parent.name
MODES = ("pairwise", "bundle")
PROJECTIONS = ("cylindrical", "planar")
MODE_LABELS = {"pairwise": "par a par", "bundle": "bundle adjustment"}
MODE_COLORS = {"pairwise": SERIES_COLORS[6], "bundle": SERIES_COLORS[3]}

BA_POINTS_PER_PAIR = 200  # default heuristic: inliers sampled per pair in the optimization (speed)
BA_HUBER_PX = 2.0  # default: Huber loss scale in px (robust to remaining outliers)
MAX_CANVAS_RATIO = 25  # default heuristic: canvas larger than 25x the mean image area = projection failed
MASK_EROSION = 2  # px removed from the border of each warped image (interpolation mixes it with black)
FOCAL_SEARCH = (0.3, 4.0)  # focal length searched between 0.3 and 4 image widths

OUTLINE_COLOR = "yellow"  # outline of the image added at each step
PREVIOUS_OUTLINE_COLOR = "white"  # outlines of the images already placed


@dataclass
class AlignedImages:
    """Every image warped onto the same canvas, in composition order (BFS from the reference)."""

    warped: list[np.ndarray]  # canvas-sized BGR images
    masks: list[np.ndarray]  # canvas-sized uint8 masks (255 = pixel comes from the image)
    indices: list[int]  # image index of each entry
    transforms: dict[int, np.ndarray]  # image k -> canvas
    parents: dict[int, int | None]  # tree neighbor already placed when k is added
    canvas_size: tuple[int, int]  # (width, height)


@dataclass
class AlignmentResult:
    aligned: AlignedImages
    images: list[np.ndarray]  # images actually warped (projected onto the cylinder when cylindrical)
    pairs: dict[tuple[int, int], PairHomography]  # pair estimates in the coordinates of `images`
    transforms: dict[str, dict[int, np.ndarray]]  # global transforms of each computed mode
    projection: str
    focal: float | None  # cylinder radius (px) when cylindrical


def oriented(pairs: dict[tuple[int, int], PairHomography], i: int, j: int) -> PairHomography:
    return pairs[i, j] if i < j else pairs[j, i].inverse()


def estimate_focal(pairs: dict, tree_edges: list[tuple[int, int]], shape: tuple) -> float:
    """For a camera that only rotates, H = K R K^-1. The focal length f (in K) is the one that makes
    K^-1 H K closest to a rotation matrix for every tree edge (grid search)."""
    h, w = shape[:2]
    homographies = [pairs[e].H for e in tree_edges]

    def rotation_error(f: float) -> float:
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        error = 0.0
        for H in homographies:
            R = np.linalg.inv(K) @ H @ K
            R /= np.cbrt(np.linalg.det(R))
            error += np.linalg.norm(R @ R.T - np.eye(3))
        return error

    candidates = np.geomspace(FOCAL_SEARCH[0] * w, FOCAL_SEARCH[1] * w, 400)
    return float(candidates[np.argmin([rotation_error(f) for f in candidates])])


def cylindrical_points(pts: np.ndarray, f: float, shape: tuple) -> np.ndarray:
    """Image point -> cylinder point: x' = f atan(x/f), y' = f y / sqrt(x^2 + f^2) (centered coordinates)."""
    h, w = shape[:2]
    x, y = pts[:, 0] - w / 2, pts[:, 1] - h / 2
    return np.c_[f * np.arctan2(x, f) + w / 2, f * y / np.hypot(x, f) + h / 2].astype(np.float32)


def cylindrical_image(image: np.ndarray, f: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse mapping of every cylinder pixel back to the image (cv2.remap), plus its validity mask."""
    h, w = image.shape[:2]
    xc, yc = np.meshgrid(np.arange(w, dtype=np.float32) - w / 2, np.arange(h, dtype=np.float32) - h / 2)
    theta = xc / f
    map_x = (f * np.tan(theta) + w / 2).astype(np.float32)
    map_y = (yc / np.cos(theta) + h / 2).astype(np.float32)
    projected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.remap(np.full((h, w), 255, np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return projected, mask


def to_cylinder(images: list[np.ndarray], pairs: dict, f: float) -> tuple[list, list, dict]:
    """Projects images and matched points onto the cylinder and re-estimates every pair with RANSAC."""
    projected = [cylindrical_image(image, f) for image in images]
    new_pairs = {}
    for (i, j), pair in pairs.items():
        pts_i = cylindrical_points(pair.pts_i, f, images[i].shape)
        pts_j = cylindrical_points(pair.pts_j, f, images[j].shape)
        new_pairs[i, j] = fit_homography(i, j, pts_i, pts_j)
    return [p[0] for p in projected], [p[1] for p in projected], new_pairs


def chain_homographies(pairs: dict, tree_edges: list[tuple[int, int]], reference: int
                       ) -> tuple[dict[int, np.ndarray], list[int], dict[int, int | None]]:
    """Pairwise mode: BFS over the tree from the reference; G_child = G_parent @ H_(child -> parent)."""
    neighbors: dict[int, list[int]] = {}
    for i, j in tree_edges:
        neighbors.setdefault(i, []).append(j)
        neighbors.setdefault(j, []).append(i)
    transforms, parents, order = {reference: np.eye(3)}, {reference: None}, [reference]
    queue = deque([reference])
    while queue:
        p = queue.popleft()
        for c in sorted(neighbors.get(p, [])):
            if c not in transforms:
                transforms[c] = transforms[p] @ oriented(pairs, c, p).H
                transforms[c] /= transforms[c][2, 2]
                parents[c] = p
                order.append(c)
                queue.append(c)
    return transforms, order, parents


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def edge_errors(pairs: dict, edges: list[tuple[int, int]], transforms: dict[int, np.ndarray]) -> list[float]:
    """Mean reprojection error (px) of each pair's inliers under the *global* transforms: H_ij = G_j^-1 G_i."""
    errors = []
    for i, j in edges:
        pair = pairs[i, j]
        H = np.linalg.inv(transforms[j]) @ transforms[i]
        errors.append(float(reprojection_errors(H, pair.pts_i[pair.inlier_mask], pair.pts_j[pair.inlier_mask]).mean()))
    return errors


def bundle_adjust(pairs: dict, overlap_edges: list[tuple[int, int]], initial: dict[int, np.ndarray],
                  reference: int) -> dict[int, np.ndarray]:
    """Extra X1: jointly refines the 8 parameters of every G_k (reference fixed to the identity) by
    minimizing the symmetric transfer error of the inliers of *every* overlapping pair (robust Huber loss)."""
    free = sorted(k for k in initial if k != reference)
    rng = np.random.default_rng(0)
    data = []
    for i, j in overlap_edges:
        pair = pairs[i, j]
        idx = np.flatnonzero(pair.inlier_mask)
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(len(idx), BA_POINTS_PER_PAIR), replace=False)
        data.append((i, j, pair.pts_i[idx].astype(np.float64), pair.pts_j[idx].astype(np.float64)))

    def unpack(x: np.ndarray) -> dict[int, np.ndarray]:
        G = {reference: np.eye(3)}
        for n, k in enumerate(free):
            G[k] = np.append(x[8 * n: 8 * n + 8], 1.0).reshape(3, 3)
        return G

    def residuals(x: np.ndarray) -> np.ndarray:
        G = unpack(x)
        out = []
        for i, j, pi, pj in data:
            H = np.linalg.solve(G[j], G[i])  # G_j^-1 G_i: image i -> image j
            out += [(project(H, pi) - pj).ravel(), (project(np.linalg.inv(H), pj) - pi).ravel()]
        return np.concatenate(out)

    x0 = np.concatenate([(initial[k] / initial[k][2, 2]).ravel()[:8] for k in free])
    result = least_squares(residuals, x0, loss="huber", f_scale=BA_HUBER_PX, x_scale="jac")
    return unpack(result.x)


def image_corners(transform: np.ndarray, shape: tuple) -> np.ndarray:
    h, w = shape[:2]
    box = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(box, transform).reshape(-1, 2)


def warp_images(images: list[np.ndarray], masks: list[np.ndarray], transforms: dict[int, np.ndarray],
                compose_order: list[int], parents: dict[int, int | None]) -> AlignedImages:
    """Translates every G_k so the whole panorama has positive coordinates and warps images + masks
    (requirement 5.2)."""
    corners = np.concatenate([image_corners(transforms[k], images[k].shape) for k in compose_order])
    (x0, y0), (x1, y1) = np.floor(corners.min(axis=0)), np.ceil(corners.max(axis=0))
    width, height = int(x1 - x0), int(y1 - y0)
    mean_area = np.mean([images[k].shape[0] * images[k].shape[1] for k in compose_order])
    if width * height > MAX_CANVAS_RATIO * mean_area:
        raise RuntimeError(f"Canvas de {width}x{height} px: a projeção distorce demais este panorama "
                           f"(campo de visão muito amplo para a projeção planar; use --projection cylindrical).")

    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    shifted = {k: T @ transforms[k] for k in compose_order}
    warped = [cv2.warpPerspective(images[k], shifted[k], (width, height), flags=cv2.INTER_LINEAR)
              for k in compose_order]
    kernel = np.ones((2 * MASK_EROSION + 1, 2 * MASK_EROSION + 1), np.uint8)
    warped_masks = [cv2.erode(cv2.warpPerspective(masks[k], shifted[k], (width, height), flags=cv2.INTER_NEAREST),
                              kernel) for k in compose_order]
    log("Passo 4", f"{len(compose_order)} imagens alinhadas em um canvas de {width}x{height} px")
    return AlignedImages(warped, warped_masks, list(compose_order), shifted, dict(parents), (width, height))


def align(images: list[np.ndarray], pairs: dict, tree_edges: list, overlap_edges: list, reference: int,
          mode: str = "pairwise", projection: str = "cylindrical", compare_modes: bool = False) -> AlignmentResult:
    """Pipeline entry point: projection, pairwise chaining (optionally refined by bundle adjustment)
    and warping. With `compare_modes` both global alignments are computed (study)."""
    if mode not in MODES or projection not in PROJECTIONS:
        raise ValueError(f"Opções válidas: modo {MODES}, projeção {PROJECTIONS}")
    masks = [np.full(image.shape[:2], 255, np.uint8) for image in images]
    focal = None
    if projection == "cylindrical":
        focal = estimate_focal(pairs, tree_edges, images[reference].shape)
        fov = np.degrees(2 * np.arctan(images[reference].shape[1] / 2 / focal))
        log("Passo 4", f"projeção cilíndrica: focal estimada {focal:.0f} px (campo de visão de {fov:.0f} graus por imagem)")
        images, masks, pairs = to_cylinder(images, pairs, focal)

    transforms = {}
    transforms["pairwise"], order, parents = chain_homographies(pairs, tree_edges, reference)
    if mode == "bundle" or compare_modes:
        start = time.perf_counter()
        transforms["bundle"] = bundle_adjust(pairs, overlap_edges, transforms["pairwise"], reference)
        log("Passo 4", f"bundle adjustment em {time.perf_counter() - start:.1f} s ({len(overlap_edges)} pares)")
    aligned = warp_images(images, masks, transforms[mode], order, parents)
    return AlignmentResult(aligned, images, pairs, transforms, projection, focal)


def error_rows(names: list[str], result: AlignmentResult, tree_edges: list, overlap_edges: list) -> list[dict]:
    """One row per overlapping pair: RANSAC statistics + reprojection error under each global alignment."""
    errors = {mode: edge_errors(result.pairs, overlap_edges, G) for mode, G in result.transforms.items()}
    rows = []
    for n, (i, j) in enumerate(overlap_edges):
        pair = result.pairs[i, j]
        row = {"par": f"{names[i]} x {names[j]}", "na_arvore": "sim" if (i, j) in tree_edges else "não",
               "matches": pair.n_matches, "inliers": pair.n_inliers, "taxa_inliers_%": 100 * pair.inlier_ratio,
               "erro_ransac_px": pair.mean_reproj_error}
        row.update({f"erro_{mode}_px": errors[mode][n] for mode in result.transforms})
        rows.append(row)
    return rows


def summary_row(mode: str, rows: list[dict]) -> dict:
    tree = [r[f"erro_{mode}_px"] for r in rows if r["na_arvore"] == "sim"]
    other = [r[f"erro_{mode}_px"] for r in rows if r["na_arvore"] == "não"]
    every = tree + other
    return {"modo": MODE_LABELS[mode], "erro_arestas_arvore_px": float(np.mean(tree)),
            "erro_outras_sobreposicoes_px": float(np.mean(other)) if other else float("nan"),
            "erro_todos_os_pares_px": float(np.mean(every)), "erro_maximo_px": float(np.max(every))}


def save_alignment_outputs(names: list[str], result: AlignmentResult, tree_edges: list, overlap_edges: list,
                           out_dir: Path) -> None:
    rows = error_rows(names, result, tree_edges, overlap_edges)
    summary = [summary_row(mode, rows) for mode in result.transforms]
    projection = f"projeção {result.projection}" + (f", focal {result.focal:.0f} px" if result.focal else "")
    print_table(rows, f"Passo 4: homografias por par (RANSAC, {projection}) e erro de reprojeção médio sob o "
                      f"alinhamento global")
    print_table(summary, "Passo 4: erro médio de reprojeção por modo de alinhamento (px)")
    write_csv(rows, out_dir / "estatisticas_homografias.csv")
    write_csv(summary, out_dir / "erro_por_modo.csv")

    for i, j in tree_edges:
        pair = result.pairs[i, j]
        plot_ransac(result.images[i], result.images[j], pair,
                    f"{names[i]} x {names[j]}: {pair.n_inliers} de {pair.n_matches} matches são inliers "
                    f"({100 * pair.inlier_ratio:.1f}%), erro de reprojeção médio {pair.mean_reproj_error:.2f} px",
                    out_dir / f"ransac_{pair_name(names, i, j)}.jpg")
    if len(result.transforms) > 1:
        plot_errors(rows, list(result.transforms), out_dir / "comparacao_alinhamento.png")
    plot_progressive(names, result, out_dir / "alinhamento_progressivo.jpg")
    plot_outlines(names, result, out_dir / "contornos_imagens.jpg")
    log("Passo 4", f"saídas salvas em {out_dir}")


def plot_errors(rows: list[dict], modes: list[str], path: Path) -> None:
    """Reprojection error of every overlapping pair under each global alignment (grouped bars)."""
    rows = sorted(rows, key=lambda r: r["na_arvore"] != "sim")  # tree edges first
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(rows)), 4.5))
    x = np.arange(len(rows))
    width = 0.8 / len(modes)
    for n, mode in enumerate(modes):
        ax.bar(x + (n - (len(modes) - 1) / 2) * width, [r[f"erro_{mode}_px"] for r in rows], width,
               color=MODE_COLORS[mode], label=MODE_LABELS[mode])
    ax.set_xticks(x, [r["par"] for r in rows], rotation=70, ha="right", fontsize=8)
    for tick, r in zip(ax.get_xticklabels(), rows):
        tick.set_fontweight("bold" if r["na_arvore"] == "sim" else "normal")
    ax.set_ylabel("erro de reprojeção médio (px)")
    ax.legend(fontsize=10)
    ax.set_title("Erro de reprojeção de cada par sob o alinhamento global (em negrito, à esquerda: arestas da árvore)", fontsize=11)
    save_figure(fig, path)


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """Outer contour of a warped mask (the true border of the image on the canvas)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea).reshape(-1, 2)


def draw_outline(ax: plt.Axes, quad: np.ndarray, color: str, width: float) -> None:
    closed = np.vstack([quad, quad[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color="black", lw=width + 1.2)
    ax.plot(closed[:, 0], closed[:, 1], color=color, lw=width)


def composite(aligned: AlignedImages, upto: int) -> np.ndarray:
    """Simple overwrite of the first `upto` warped images (visualization only, no blending)."""
    canvas = np.zeros_like(aligned.warped[0])
    for img, mask in zip(aligned.warped[:upto], aligned.masks[:upto]):
        canvas[mask > 0] = img[mask > 0]
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def plot_progressive(names: list[str], result: AlignmentResult, path: Path) -> None:
    """Requirement 5.4: the canvas after each image is added (newest image outlined)."""
    aligned = result.aligned
    n = len(aligned.indices)
    w, h = aligned.canvas_size
    cols = 2 if w / h > 2 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * (18 / cols) * h / w + rows * 0.5))
    for step, ax in enumerate(np.atleast_1d(axes).flat):
        ax.axis("off")
        if step >= n:
            continue
        ax.imshow(composite(aligned, step + 1))
        for prev in range(step):
            draw_outline(ax, mask_outline(aligned.masks[prev]), PREVIOUS_OUTLINE_COLOR, 0.6)
        k = aligned.indices[step]
        draw_outline(ax, mask_outline(aligned.masks[step]), OUTLINE_COLOR, 1.5)
        parent = aligned.parents[k]
        suffix = " (referência)" if parent is None else f", ligada a {names[parent]}"
        ax.set_title(f"Passo {step + 1}: + {names[k]}{suffix}", fontsize=IMAGE_TITLE)
    fig.suptitle("Alinhamento progressivo (contorno amarelo = imagem adicionada no passo)", fontsize=IMAGE_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))  # room for the suptitle
    save_figure(fig, path)


def plot_outlines(names: list[str], result: AlignmentResult, path: Path) -> None:
    """Final canvas with the outline of every warped image and its composition step."""
    aligned = result.aligned
    w, h = aligned.canvas_size
    fig, ax = plt.subplots(figsize=(18, 18 * h / w + 0.8))
    ax.imshow(composite(aligned, len(aligned.indices)))
    for step, k in enumerate(aligned.indices):
        outline = mask_outline(aligned.masks[step])
        draw_outline(ax, outline, OUTLINE_COLOR, 1.2)
        (x0, y0), (x1, y1) = outline.min(axis=0), outline.max(axis=0)
        cx, cy = (x0 + x1) / 2, y0 + (y1 - y0) * (0.3 if step % 2 else 0.7)  # alternate heights: no overlap
        ax.text(cx, cy, f"{step + 1}\n{names[k]}", ha="center", va="center", fontsize=IMAGE_TEXT, color="black",
                bbox={"fc": OUTLINE_COLOR, "ec": "black", "alpha": 0.85, "pad": 2})
    ax.axis("off")
    ax.set_title(f"Imagens alinhadas no canvas ({w}x{h} px, projeção {result.projection}); "
                 f"número = ordem de composição a partir da referência", fontsize=IMAGE_TITLE, color=INK_SECONDARY)
    save_figure(fig, path)
