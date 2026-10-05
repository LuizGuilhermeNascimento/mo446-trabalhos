"""Evaluation of the composition.

Metrics (terminal + CSV):
- ghosts (%): pixels where the aligned sources disagree in color (moving object or parallax) and the
  panorama matches none of them, i.e. it shows a mix of different contents. A hard seam scores 0 by definition.
  Color (max over B, G, R) and not gray: a pink paraglider over a blue sky has almost the same gray level.
- seam discontinuity: intensity jump across the seam minus the jump that already exists in the source
  image at that spot (gray levels; 0 = invisible seam). Hard seams through objects score high.
- photometric error per neighbor pair: mean |I_a - I_b| where both aligned images overlap (misalignment).
- geometric distortion per image: warped area / original area and max deviation of the corner angles.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import (IMAGE_SUPTITLE, IMAGE_TEXT, IMAGE_TITLE, INK, INK_SECONDARY, SERIES_COLORS, log, print_table,
                    save_figure, save_image, write_csv)
from step4_homography_connection.alignment import AlignedImages, AlignmentResult, image_corners
from step5_panorama_composition.compositor import CompositionResult, PanoramaCompositor

STAGE = Path(__file__).resolve().parent.name
GHOST_COLOR = "red"  # ghost pixels of the naive composition drawn over the panorama
SEAM_COLOR = "red"  # seam lines drawn over the panorama
CROP_COLOR = "yellow"  # rectangles of the zoomed regions
DISAGREEMENT = 30  # default: intensity levels (any color channel) for two aligned sources to "disagree"
GHOST_REGIONS = 4  # regions zoomed in the deghosting figure
MAX_REGION = 0.4  # default: a ghost region is compact if its box is at most 40% of the panorama height
MIN_CROP = 160  # px, smallest zoomed region
GHOST_MERGE = 31  # default: px of dilation that joins the copies of the same moving object into one region

STUDY = [("none", "none"), ("none", "feather"), ("seam", "none"), ("seam", "feather"), ("seam", "multiband")]
METHOD_LABELS = {"none_none": "Média simples", "none_feather": "Feathering",
                 "seam_none": "Costura", "seam_feather": "Costura + feathering",
                 "seam_multiband": "Costura + multibanda"}


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- metrics (6.5)

def sources(aligned: AlignedImages) -> tuple[np.ndarray, np.ndarray]:
    """Gray levels (N x H x W) and validity (N x H x W) of the aligned images."""
    return np.stack([gray(w) for w in aligned.warped]), np.stack([m > 0 for m in aligned.masks])


def ghost_mask(aligned: AlignedImages, panorama: np.ndarray, with_spread: bool = False):
    """(ghost pixels, pixels covered by 2+ images[, color disagreement]), comparing colors channel by channel."""
    colors = np.stack(aligned.warped).astype(np.int16)  # N x H x W x 3
    valid = np.stack([m > 0 for m in aligned.masks])[..., None]
    overlap = valid[..., 0].sum(axis=0) >= 2
    spread = (np.where(valid, colors, -1).max(axis=0) - np.where(valid, colors, 256).min(axis=0)).max(axis=-1)
    distance = np.abs(colors - panorama.astype(np.int16)).max(axis=-1)  # N x H x W
    closest = np.where(valid[..., 0], distance, 256).min(axis=0)
    ghosts = overlap & (spread > DISAGREEMENT) & (closest > DISAGREEMENT / 2)
    return (ghosts, overlap, spread) if with_spread else (ghosts, overlap)


def ghost_percent(aligned: AlignedImages, panorama: np.ndarray) -> float:
    ghosts, overlap = ghost_mask(aligned, panorama)
    return float(100 * ghosts.sum() / max(overlap.sum(), 1))


def seam_discontinuity(aligned: AlignedImages, panorama: np.ndarray, labels: np.ndarray | None) -> float:
    """Mean excess jump (gray levels) across the seams of `labels`: |P(p) - P(q)| minus the jump
    |I_a(p) - I_a(q)| of the source image a of p, over neighbor pairs (p, q) split by a seam."""
    if labels is None:
        return float("nan")
    g, valid = sources(aligned)
    pano = gray(panorama)
    excess = []
    for dy, dx in ((0, 1), (1, 0)):  # right and bottom neighbors
        h, w = labels.shape
        a, b = labels[:h - dy, :w - dx], labels[dy:, dx:]
        ys, xs = np.nonzero((a != b) & (a >= 0) & (b >= 0))
        src = a[ys, xs]
        both_sides = valid[src, ys, xs] & valid[src, ys + dy, xs + dx]  # source a covers p and q
        ys, xs, src = ys[both_sides], xs[both_sides], src[both_sides]
        step_panorama = np.abs(pano[ys, xs] - pano[ys + dy, xs + dx])
        step_source = np.abs(g[src, ys, xs] - g[src, ys + dy, xs + dx])
        excess.append(np.clip(step_panorama - step_source, 0, None))
    values = np.concatenate(excess)
    return float(values.mean()) if len(values) else float("nan")


def method_row(name: str, result: CompositionResult, seam_labels: np.ndarray | None) -> dict:
    """`seam_labels` locates the seams where every method is compared (the same pixels for all)."""
    return {"metodo": METHOD_LABELS.get(name, name), "fantasmas_%": ghost_percent(result.aligned, result.panorama),
            "descontinuidade_costura": seam_discontinuity(result.aligned, result.panorama, seam_labels),
            "tempo_ms": result.time_ms}


def overlap_rows(names: list[str], aligned: AlignedImages) -> list[dict]:
    """Photometric error of each image with its tree neighbor, where both overlap."""
    position = {k: n for n, k in enumerate(aligned.indices)}
    rows = []
    for n, k in enumerate(aligned.indices):
        parent = aligned.parents[k]
        if parent is None:
            continue
        p = position[parent]
        both = (aligned.masks[n] > 0) & (aligned.masks[p] > 0)
        if not both.any():
            continue
        error = np.abs(gray(aligned.warped[n]) - gray(aligned.warped[p]))[both]
        rows.append({"par": f"{names[k]} x {names[parent]}", "pixels_sobrepostos": int(both.sum()),
                     "erro_fotometrico_medio": float(error.mean()),
                     "pixels_discordantes_%": float(100 * (error > DISAGREEMENT).mean())})
    return rows


def corner_angles(quad: np.ndarray) -> np.ndarray:
    angles = []
    for c in range(4):
        a, b = quad[c - 1] - quad[c], quad[(c + 1) % 4] - quad[c]
        angles.append(np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))))
    return np.array(angles)


def distortion_rows(names: list[str], alignment: AlignmentResult) -> list[dict]:
    aligned = alignment.aligned
    rows = []
    for n, k in enumerate(aligned.indices):
        h, w = alignment.images[k].shape[:2]
        quad = image_corners(aligned.transforms[k], alignment.images[k].shape)
        rows.append({"imagem": names[k], "razao_area": float((aligned.masks[n] > 0).sum() / (h * w)),
                     "desvio_angulo_max_graus": float(np.abs(corner_angles(quad) - 90).max()),
                     "largura_px": int(np.ptp(quad[:, 0])), "altura_px": int(np.ptp(quad[:, 1]))})
    return rows


def ghost_regions(result: CompositionResult, crop: tuple | None = None) -> list[tuple[int, int, int, int]]:
    """Square regions (x, y, w, h) around the ghosts of the naive composition. Nearby copies of the same
    object are merged (dilation), large blobs are skipped, and interior blobs come first, ordered by how
    strongly the sources disagree: moving objects (e.g. the paraglider over the sky) disagree strongly
    away from the borders, while parallax of nearby trees and rocks touches the panorama border.
    `crop` (x, y, w, h) is added first when given."""
    ghosts, _, spread = ghost_mask(result.aligned, result.naive, with_spread=True)
    h, w = ghosts.shape
    blobs = cv2.dilate(ghosts.astype(np.uint8), np.ones((GHOST_MERGE, GHOST_MERGE), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blobs)

    def priority(k: int) -> tuple[bool, float]:
        x, y, bw, bh = stats[k, :4]
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        return touches_border, -float(spread[(labels == k) & ghosts].mean())

    compact = sorted((k for k in range(1, n) if max(stats[k, 2], stats[k, 3]) <= MAX_REGION * h), key=priority)
    regions = [tuple(crop)] if crop else []
    for k in compact:
        x, y, bw, bh = stats[k, :4]
        size = int(min(max(MIN_CROP, 1.5 * max(bw, bh)), h, w))
        rx = int(np.clip(x + bw / 2 - size / 2, 0, w - size))
        ry = int(np.clip(y + bh / 2 - size / 2, 0, h - size))
        if all(abs(rx - ox) > size / 2 or abs(ry - oy) > size / 2 for ox, oy, *_ in regions):  # no repeats
            regions.append((rx, ry, size, size))
        if len(regions) == GHOST_REGIONS:
            break
    return regions


def label_inside(ax: plt.Axes, text: str) -> None:
    """Label written inside the top-left corner of an image panel."""
    ax.text(0.02, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=IMAGE_TEXT, color=INK,
            bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 3})


def plot_region_comparison(panorama: np.ndarray, crops: list[tuple[str, np.ndarray]], region: tuple, title: str,
                           path: Path, ncols: int | None = None) -> None:
    """Top: panorama with the region outlined. Below: the region of each version, enlarged, labeled inside."""
    x, y, w, h = region
    ph, pw = panorama.shape[:2]
    ncols = ncols or len(crops)
    nrows = -(-len(crops) // ncols)
    crop_w = 18 / ncols
    fig = plt.figure(figsize=(18, 18 * ph / pw + nrows * crop_w * h / w + 1.0), layout="constrained")
    grid = fig.add_gridspec(1 + nrows, 2 * ncols, height_ratios=[18 * ph / pw] + [crop_w * h / w] * nrows)
    top = fig.add_subplot(grid[0, :])
    top.imshow(rgb(panorama))
    top.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=CROP_COLOR, lw=2.5))
    top.axis("off")
    for n, (label, image) in enumerate(crops):
        r, c = divmod(n, ncols)
        start = 2 * c + (ncols - min(ncols, len(crops) - r * ncols))
        ax = fig.add_subplot(grid[1 + r, start:start + 2])
        ax.imshow(rgb(image[y:y + h, x:x + w]), interpolation="lanczos")
        label_inside(ax, label)
        ax.axis("off")
    fig.suptitle(title, fontsize=IMAGE_SUPTITLE)
    save_figure(fig, path)


def plot_deghosting(result: CompositionResult, method: str, region: tuple, group: str, path: Path) -> None:
    """Requirement 6.4: the same region without and with ghost removal, enlarged side by side."""
    name = METHOD_LABELS.get(method, method)
    plot_region_comparison(result.panorama, [("Sem remoção de fantasmas (média simples)", result.naive),
                                             (f"Com remoção de fantasmas ({name[0].lower() + name[1:]})",
                                              result.panorama)],
                           region, f"Remoção de Fantasmas: Conjunto {group}", path, ncols=3)  # 3 columns: smaller crops


def seam_pixels(labels: np.ndarray) -> np.ndarray:
    seams = np.zeros(labels.shape, bool)
    seams[:, :-1] |= (labels[:, :-1] != labels[:, 1:]) & (labels[:, :-1] >= 0) & (labels[:, 1:] >= 0)
    seams[:-1] |= (labels[:-1] != labels[1:]) & (labels[:-1] >= 0) & (labels[1:] >= 0)
    return seams


def plot_seams(names: list[str], result: CompositionResult, group: str, path: Path) -> None:
    """Which image supplies each pixel after the optimal seams (number = composition order)."""
    labels, aligned = result.labels, result.aligned
    ph, pw = labels.shape
    colors = np.array([plt.cm.tab10(n % 10)[:3] for n in range(len(aligned.indices))]) * 255
    label_map = np.zeros((ph, pw, 3), np.float32)
    label_map[labels >= 0] = colors[labels[labels >= 0]]
    overlay = (0.45 * rgb(result.panorama) + 0.55 * label_map).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(18, 18 * ph / pw + 0.8), layout="constrained")
    ax.imshow(overlay)
    for n, k in enumerate(aligned.indices):
        region = (labels == n).astype(np.uint8)
        if region.any():  # label at the most interior point of the region (never outside thin strips)
            depth = cv2.distanceTransform(np.pad(region, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]  # image border = limit
            yy, xx = np.unravel_index(np.argmax(depth), depth.shape)
            ax.text(xx, yy, f"{n + 1}\n{names[k]}", ha="center", va="center",
                    fontsize=IMAGE_TEXT, bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 2})
    ax.set_xlim(-0.5, pw - 0.5)
    ax.set_ylim(ph - 0.5, -0.5)
    ax.axis("off")
    ax.set_title(f"Imagem de Origem de Cada Pixel após a Costura Ótima: Conjunto {group}", fontsize=IMAGE_SUPTITLE)
    save_figure(fig, path)


def plot_methods(results: dict[str, CompositionResult], region: tuple, group: str, path: Path) -> None:
    """Study: the same region (a seam) in every composition method, enlarged."""
    plot_region_comparison(results["seam_feather"].panorama,
                           [(METHOD_LABELS.get(name, name), r.panorama) for name, r in results.items()],
                           region, f"Métodos de Composição na Mesma Região: Conjunto {group}", path, ncols=3)


def plot_metrics(rows: list[dict], group: str, path: Path) -> None:
    """One panel per metric; one horizontal bar per composition method."""
    names = [r["metodo"] for r in rows]
    colors = [SERIES_COLORS[n % len(SERIES_COLORS)] for n in range(len(rows))]
    panels = [("fantasmas_%", "Pixels com Fantasma", "Pixels sobrepostos com fantasma (%)"),
              ("descontinuidade_costura", "Descontinuidade na Costura", "Salto de intensidade (níveis de cinza)")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), layout="constrained")
    for ax, (key, title, xlabel) in zip(axes, panels):
        ax.barh(names, [0.0 if np.isnan(r[key]) else r[key] for r in rows], height=0.6, color=colors)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e1e0d9", lw=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=12)
    axes[0].set_ylabel("Método de composição", fontsize=10)
    axes[1].tick_params(labelleft=False)
    fig.suptitle(f"Métricas dos Métodos de Composição: Conjunto {group}", fontsize=14)
    save_figure(fig, path)


def save_composition_outputs(names: list[str], alignment: AlignmentResult, method: str, result: CompositionResult,
                             group: str, out_dir: Path, region: tuple[int, int, int, int] | None = None) -> None:
    """`region` (x, y, w, h in the cropped panorama) is shown with and without ghost removal; without it,
    the region with the strongest ghosts is chosen automatically."""
    naive = CompositionResult(result.naive, result.naive, None, result.mask, result.aligned, result.rect, 0.0)
    rows = [method_row("none_none", naive, result.labels), method_row(method, result, result.labels)]
    print_table(rows, "Passo 5: composição sem e com remoção de fantasmas")
    overlaps = overlap_rows(names, result.aligned)
    print_table(overlaps, "Passo 5: erro fotométrico entre imagens vizinhas na sobreposição (níveis de cinza)")
    distortion = distortion_rows(names, alignment)
    print_table(distortion, "Passo 5: distorção geométrica de cada imagem no panorama")
    write_csv(rows, out_dir / "metricas_composicao.csv")
    write_csv(overlaps, out_dir / "erro_fotometrico_sobreposicao.csv")
    write_csv(distortion, out_dir / "distorcao_por_imagem.csv")

    plot_deghosting(result, method, region or default_region(result), group, out_dir / "comparacao_deghosting.jpg")
    if result.labels is not None:
        plot_seams(names, result, group, out_dir / "costuras.jpg")
    log("Passo 5", f"saídas salvas em {out_dir}")


def default_region(result: CompositionResult) -> tuple[int, int, int, int]:
    regions = ghost_regions(result)
    h, w = result.panorama.shape[:2]
    return regions[0] if regions else (0, 0, min(h, w), min(h, w))


def compare_compositions(alignment: AlignmentResult, region: tuple[int, int, int, int] | None, group: str,
                         out_dir: Path) -> None:
    """Study: every deghosting x blending combination, measured across the same seams and shown on the
    same enlarged region (`region`, e.g. a seam through a nearby object; automatic when None)."""
    results = {f"{d}_{b}": PanoramaCompositor(d, b).compose(alignment.aligned) for d, b in STUDY}
    seams = results["seam_feather"].labels  # every method is measured across the same seams
    rows = [method_row(name, r, seams) for name, r in results.items()]
    print_table(rows, "Passo 5: comparação dos métodos de composição")
    write_csv(rows, out_dir / "metricas_metodos_composicao.csv")
    plot_metrics(rows, group, out_dir / "metricas_metodos_composicao.png")
    plot_methods(results, region or default_region(results["seam_feather"]), group,
                 out_dir / "comparacao_metodos.jpg")
