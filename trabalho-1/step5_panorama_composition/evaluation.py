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

from common import (IMAGE_SUPTITLE, IMAGE_TEXT, IMAGE_TITLE, INK_SECONDARY, SERIES_COLORS, log, print_table,
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
METHOD_LABELS = {"none_none": "média (ingênua)", "none_feather": "feathering sem deghosting",
                 "seam_none": "costura sem blending", "seam_feather": "costura + feathering",
                 "seam_multiband": "costura + multibanda"}


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


def plot_deghosting(result: CompositionResult, method: str, regions: list[tuple], path: Path) -> None:
    """Requirement 6.4: ghost map of the naive composition and the same regions without / with deghosting."""
    ph, pw = result.panorama.shape[:2]
    ghosts, _ = ghost_mask(result.aligned, result.naive)
    n_cols = max(len(regions), 1)
    crop_w = 18 / n_cols
    fig = plt.figure(figsize=(18, 18 * ph / pw + 2 * crop_w + 2.5), layout="constrained")
    grid = fig.add_gridspec(3, n_cols, height_ratios=[18 * ph / pw, crop_w, crop_w])
    top = fig.add_subplot(grid[0, :])
    top.imshow(rgb(result.panorama))
    ys, xs = np.nonzero(ghosts)
    top.scatter(xs, ys, s=0.2, c=GHOST_COLOR, linewidths=0, alpha=0.5)
    for n, (x, y, w, h) in enumerate(regions):
        top.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=CROP_COLOR, lw=2))
        top.text(x, y - 6, str(n + 1), color="black", fontsize=IMAGE_TEXT, fontweight="bold",
                 bbox={"fc": CROP_COLOR, "ec": "none", "pad": 1.5})
    top.set_title(f"Panorama final ({METHOD_LABELS.get(method, method)}); vermelho = pixels com fantasma na "
                  f"composição ingênua; amarelo = regiões ampliadas abaixo", fontsize=IMAGE_TITLE)
    top.axis("off")
    for n, (x, y, w, h) in enumerate(regions):
        for row, (image, label) in enumerate([(result.naive, "sem deghosting"), (result.panorama, "com deghosting")]):
            ax = fig.add_subplot(grid[1 + row, n])
            ax.imshow(rgb(image[y:y + h, x:x + w]), interpolation="nearest")
            ax.set_title(f"Região {n + 1}: {label}", fontsize=IMAGE_TITLE)
            ax.axis("off")
    fig.suptitle(f"Remoção de fantasmas: mesmas regiões sem deghosting (média ingênua, meio) e com deghosting "
                 f"({METHOD_LABELS.get(method, method)}, embaixo)", fontsize=IMAGE_SUPTITLE)
    save_figure(fig, path)


def seam_pixels(labels: np.ndarray) -> np.ndarray:
    seams = np.zeros(labels.shape, bool)
    seams[:, :-1] |= (labels[:, :-1] != labels[:, 1:]) & (labels[:, :-1] >= 0) & (labels[:, 1:] >= 0)
    seams[:-1] |= (labels[:-1] != labels[1:]) & (labels[:-1] >= 0) & (labels[1:] >= 0)
    return seams


def plot_seams(names: list[str], result: CompositionResult, path: Path) -> None:
    """Top: which image supplies each pixel. Bottom: panorama with the seams."""
    labels, aligned = result.labels, result.aligned
    ph, pw = labels.shape
    colors = np.array([plt.cm.tab10(n % 10)[:3] for n in range(len(aligned.indices))]) * 255
    label_map = np.zeros((ph, pw, 3), np.float32)
    label_map[labels >= 0] = colors[labels[labels >= 0]]
    overlay = (0.45 * rgb(result.panorama) + 0.55 * label_map).astype(np.uint8)
    fig, axes = plt.subplots(2, 1, figsize=(18, 2 * 18 * ph / pw + 2), layout="constrained")
    axes[0].imshow(overlay)
    for n, k in enumerate(aligned.indices):
        ys, xs = np.nonzero(labels == n)
        if len(xs):
            axes[0].text(np.median(xs), np.median(ys), f"{n + 1}\n{names[k]}", ha="center", va="center",
                         fontsize=IMAGE_TEXT, bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 2})
    axes[0].set_title("Imagem-fonte de cada pixel após a costura ótima (número = ordem de composição)",
                      fontsize=IMAGE_TITLE)
    axes[1].imshow(rgb(result.panorama))
    ys, xs = np.nonzero(seam_pixels(labels))
    axes[1].scatter(xs, ys, s=0.4, c=SEAM_COLOR, linewidths=0)
    axes[1].set_title("Costuras ótimas (vermelho) sobre o panorama final", fontsize=IMAGE_TITLE)
    for ax in axes:
        ax.axis("off")
    save_figure(fig, path)


def plot_methods(results: dict[str, CompositionResult], rows: list[dict], region: tuple, path: Path) -> None:
    """Study: every method on the full panorama (left) and on the same zoomed region (right)."""
    x, y, w, h = region
    ph, pw = next(iter(results.values())).panorama.shape[:2]
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(20, n * 20 / (pw / ph + 1) + 1.5), layout="constrained",
                             gridspec_kw={"width_ratios": [pw / ph, 1]})
    for (name, result), row, (left, right) in zip(results.items(), rows, axes):
        left.imshow(rgb(result.panorama))
        left.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=CROP_COLOR, lw=2))
        discontinuity = row["descontinuidade_costura"]
        left.set_title(f"{METHOD_LABELS.get(name, name)}: fantasmas {row['fantasmas_%']:.2f}%, descontinuidade "
                       f"na costura {discontinuity:.2f}", fontsize=IMAGE_TITLE)
        right.imshow(rgb(result.panorama[y:y + h, x:x + w]), interpolation="nearest")
        right.set_title("região ampliada", fontsize=IMAGE_TITLE)
        left.axis("off")
        right.axis("off")
    fig.suptitle("Métodos de composição: panorama completo (esquerda) e a mesma região ampliada (direita)",
                 fontsize=IMAGE_SUPTITLE)
    save_figure(fig, path)


def plot_metrics(rows: list[dict], path: Path) -> None:
    """Small multiples, one metric per panel: ghosts and seam discontinuity per method."""
    names = [r["metodo"] for r in rows]
    colors = [SERIES_COLORS[n % len(SERIES_COLORS)] for n in range(len(rows))]
    panels = [("fantasmas_%", "Fantasmas (% dos pixels sobrepostos)\nmenor = menos fantasmas"),
              ("descontinuidade_costura", "Descontinuidade na costura (níveis de cinza)\n0 = costura invisível")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), layout="constrained")
    for ax, (key, title) in zip(axes, panels):
        values = [r[key] for r in rows]
        ax.barh(names, [0 if np.isnan(v) else v for v in values], color=colors, height=0.6)
        for yy, v in enumerate(values):
            ax.text(0 if np.isnan(v) else v, yy, "  -" if np.isnan(v) else f"  {v:.2f}", va="center",
                    fontsize=10, color=INK_SECONDARY)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e1e0d9", lw=0.6)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25)
        ax.set_title(title, fontsize=11)
    axes[1].tick_params(labelleft=False)
    fig.suptitle("Métricas dos métodos de composição", fontsize=12)
    save_figure(fig, path)


def save_composition_outputs(names: list[str], alignment: AlignmentResult, method: str, result: CompositionResult,
                             out_dir: Path, crop: tuple[int, int, int, int] | None = None) -> list[tuple]:
    naive = CompositionResult(result.naive, result.naive, None, result.mask, result.aligned, result.rect, 0.0)
    rows = [method_row("none_none", naive, result.labels), method_row(method, result, result.labels)]
    print_table(rows, "Passo 5: composição sem e com deghosting")
    overlaps = overlap_rows(names, result.aligned)
    print_table(overlaps, "Passo 5: erro fotométrico entre imagens vizinhas na sobreposição (níveis de cinza)")
    distortion = distortion_rows(names, alignment)
    print_table(distortion, "Passo 5: distorção geométrica de cada imagem no panorama")
    write_csv(rows, out_dir / "metricas_composicao.csv")
    write_csv(overlaps, out_dir / "erro_fotometrico_sobreposicao.csv")
    write_csv(distortion, out_dir / "distorcao_por_imagem.csv")

    regions = ghost_regions(result, crop)
    plot_deghosting(result, method, regions, out_dir / "comparacao_deghosting.jpg")
    save_image(out_dir / f"panorama_{method}.jpg", result.panorama)
    if result.labels is not None:
        plot_seams(names, result, out_dir / "costuras.jpg")
    log("Passo 5", f"saídas salvas em {out_dir}")
    return regions


def compare_compositions(alignment: AlignmentResult, region: tuple[int, int, int, int] | None, out_dir: Path) -> None:
    """Study: every deghosting x blending combination, measured across the same seams and shown on the
    same zoomed region (the first region of the deghosting figure)."""
    results = {f"{d}_{b}": PanoramaCompositor(d, b).compose(alignment.aligned) for d, b in STUDY}
    seams = results["seam_feather"].labels  # every method is measured across the same seams
    rows = [method_row(name, r, seams) for name, r in results.items()]
    print_table(rows, "Passo 5: comparação dos métodos de composição")
    write_csv(rows, out_dir / "comparacao_metodos.csv")
    plot_metrics(rows, out_dir / "comparacao_metodos.png")
    h, w = results["seam_feather"].panorama.shape[:2]
    plot_methods(results, rows, region or (0, 0, min(h, w), min(h, w)), out_dir / "comparacao_metodos.jpg")
