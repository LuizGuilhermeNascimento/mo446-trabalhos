from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from common import (IMAGE_SUPTITLE, IMAGE_TEXT, IMAGE_TITLE, INK_SECONDARY, ImageSet, log, pair_name,
                    print_table, save_figure, write_csv)
from step1_key_points_detection.detector import DETECTOR_COLORS, METHODS as DETECTORS, Features

STAGE = Path(__file__).resolve().parent.name
MATCHERS = ("BF", "FLANN")

RATIO = 0.75  # Lowe's ratio threshold; usual default (Lowe 2004 suggests 0.8, OpenCV tutorials use 0.7-0.75)
RATIOS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)  # thresholds of the ratio test sweep
FLANN_KDTREE = {"algorithm": 1, "trees": 5}  # float descriptors (SIFT); OpenCV tutorial default
FLANN_LSH = {"algorithm": 6, "table_number": 6, "key_size": 12, "multi_probe_level": 1}  # binary; OpenCV tutorial default
FLANN_SEARCH = {"checks": 50}  # OpenCV tutorial default
RANSAC_THRESHOLD = 3.0  # px; OpenCV's default ransacReprojThreshold. Only used here to *evaluate* matches

CANDIDATE_COLOR = "yellow"  # matches before the ratio test
ACCEPTED_COLOR = "lime"  # matches accepted by the ratio test (solid lines)
REJECTED_COLOR = "red"  # matches discarded by the ratio test (dashed lines)
MAX_LINES = 150  # lines drawn per category, so the figures stay readable


@dataclass
class MatchResult:
    """Matches from image A (query) to image B (train)."""

    knn: list[tuple[cv2.DMatch, ...]]  # the 2 nearest neighbors in B of each descriptor of A
    good: list[cv2.DMatch]  # accepted by the ratio test
    rejected: list[cv2.DMatch]  # nearest neighbor discarded by the ratio test
    ratio: float
    method: str
    time_ms: float  # knnMatch time

    @property
    def candidates(self) -> list[cv2.DMatch]:
        """Nearest neighbor of every descriptor: the matches *before* the ratio test."""
        return [pair[0] for pair in self.knn if pair]


def ratio_test(knn: list[tuple[cv2.DMatch, ...]], ratio: float) -> tuple[list, list]:
    """Lowe: keep m only if d(m) < ratio * d(second neighbor). Without a second neighbor it is rejected."""
    good, rejected = [], []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
        elif pair:
            rejected.append(pair[0])
    return good, rejected


class FeatureMatcher:
    """Brute Force or FLANN matcher; norm / index chosen by the descriptor type (float or binary)."""

    def __init__(self, method: str = "FLANN", ratio: float = RATIO):
        self.method = method.upper()
        if self.method not in MATCHERS:
            raise ValueError(f"Matcher desconhecido: {method}. Opções: {', '.join(MATCHERS)}")
        self.ratio = ratio

    def _backend(self, binary: bool):
        if self.method == "BF":
            return cv2.BFMatcher(cv2.NORM_HAMMING if binary else cv2.NORM_L2)
        return cv2.FlannBasedMatcher(FLANN_LSH if binary else FLANN_KDTREE, FLANN_SEARCH)

    def match(self, feat_a: Features, feat_b: Features) -> MatchResult:
        if len(feat_a) < 2 or len(feat_b) < 2:
            return MatchResult([], [], [], self.ratio, self.method, 0.0)
        backend = self._backend(feat_a.is_binary)
        start = time.perf_counter()
        knn = backend.knnMatch(feat_a.descriptors, feat_b.descriptors, k=2)
        time_ms = (time.perf_counter() - start) * 1000
        good, rejected = ratio_test(knn, self.ratio)
        return MatchResult(list(knn), good, rejected, self.ratio, self.method, time_ms)


def geometric_inliers(feat_a: Features, feat_b: Features, matches: list[cv2.DMatch]) -> np.ndarray:
    """Boolean mask of the matches consistent with a RANSAC homography (evaluation only)."""
    if len(matches) < 4:
        return np.zeros(len(matches), bool)
    pts_a = feat_a.points[[m.queryIdx for m in matches]]
    pts_b = feat_b.points[[m.trainIdx for m in matches]]
    _, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, RANSAC_THRESHOLD)
    return np.zeros(len(matches), bool) if mask is None else mask.ravel().astype(bool)


def pair_stats(names: list[str], i: int, j: int, feat_a: Features, feat_b: Features, result: MatchResult) -> dict:
    inliers = int(geometric_inliers(feat_a, feat_b, result.good).sum())
    accepted = len(result.good)
    return {"detector": feat_a.method, "matcher": result.method, "par": f"{names[i]} x {names[j]}",
            "candidatos": len(result.candidates), "aceitos": accepted,
            "taxa_aceitacao_%": 100 * accepted / max(len(result.candidates), 1),
            "inliers": inliers, "precisao_%": 100 * inliers / max(accepted, 1), "tempo_ms": result.time_ms}


def compare_matchers(image_set: ImageSet, out_dir: Path, features: dict[str, list[Features]],
                     overlapping: list[tuple[int, int]], neighbors: list[tuple[int, int]], ratio: float = RATIO,
                     detector: str = "SIFT", matcher: str = "FLANN") -> list[dict]:
    """Matches every pair of images with every detector x matcher combination and compares them.

    `features` holds the keypoints of every detector (step 1); `overlapping` (pairs that overlap) and
    `neighbors` (consecutive pairs of the sequence) come from the step 3 graph. `detector` + `matcher` is the combination drawn in the figures and used in the ratio sweep.
    """
    names = image_set.names
    pairs = list(itertools.combinations(range(len(image_set)), 2))

    results, rows = {}, {}
    for d, m in itertools.product(DETECTORS, MATCHERS):
        fm = FeatureMatcher(m, ratio)
        fm.match(features[d][0], features[d][1])  # warm-up, not timed
        results[d, m] = {(i, j): fm.match(features[d][i], features[d][j]) for i, j in pairs}
        rows[d, m] = {(i, j): pair_stats(names, i, j, features[d][i], features[d][j], r)
                      for (i, j), r in results[d, m].items()}
        log("Passo 2", f"{d} + {m}: {len(pairs)} pares emparelhados")

    separate = [p for p in pairs if p not in overlapping]
    log("Passo 2", f"{len(overlapping)} pares com sobreposição e {len(separate)} sem (grafo do passo 3)")

    summary = [summarize(d, m, rows, results, overlapping, separate) for d, m in results]
    print_table([{"combinação": f"{r['detector']} + {r['matcher']}", "tempo (ms)": f"{r['tempo_ms']:.1f}",
                  "aceitos": f"{r['aceitos']:.0f}", "aceitação (%)": f"{r['taxa_aceitacao_%']:.1f}",
                  "inliers": f"{r['inliers']:.0f}", "precisão (%)": f"{r['precisao_%']:.1f}",
                  "concordância c/ BF (%)": f"{r['concordancia_BF_%']:.1f}",
                  "falsos aceitos (sem sobrep.)": f"{r['falsos_aceitos']:.1f}"} for r in summary],
                f"Passo 2: comparação detector x matcher (média sobre {len(overlapping)} pares com sobreposição, "
                f"ratio test {ratio})")
    sweep = ratio_sweep(features, results, overlapping, matcher)
    print_table([r for r in sweep if r["detector"] == detector],
                f"Passo 2: curva do ratio test ({detector} + {matcher}, média sobre os pares com sobreposição)")

    write_csv(summary, out_dir / "comparacao_matchers.csv")
    write_csv([r for combo in rows.values() for r in combo.values()], out_dir / "comparacao_matchers_por_par.csv")
    write_csv(sweep, out_dir / "curva_ratio_test.csv")
    plot_metrics(summary, len(overlapping), out_dir / "comparacao_matchers.png")
    plot_ratio_sweep(sweep, matcher, ratio, out_dir / "curva_ratio_test.png")

    feats = features[detector]
    for i, j in neighbors:
        plot_before_after(image_set, i, j, feats[i], feats[j], results[detector, matcher][i, j],
                          rows[detector, matcher][i, j], out_dir / f"matches_{pair_name(names, i, j)}.jpg")
    if overlapping:  # representative pair: median number of inliers
        pair = sorted(overlapping, key=lambda p: rows["SIFT", "BF"][p]["inliers"])[len(overlapping) // 2]
        plot_combinations(image_set, pair, features, results, rows,
                          out_dir / f"comparacao_matchers_{pair_name(names, *pair)}.jpg")
    if separate:  # false matches: the non-overlapping pair with most accepted matches (requirement 3.4)
        i, j = max(separate, key=lambda p: rows[detector, matcher][p]["aceitos"])
        plot_before_after(image_set, i, j, feats[i], feats[j], results[detector, matcher][i, j],
                          rows[detector, matcher][i, j], out_dir / f"matches_sem_sobreposicao_{pair_name(names, i, j)}.jpg")
    log("Passo 2", f"estudo comparativo salvo em {out_dir}")
    return summary


def summarize(d: str, m: str, rows: dict, results: dict, overlapping: list, separate: list) -> dict:
    over = [rows[d, m][p] for p in overlapping]
    mean = lambda key: float(np.mean([r[key] for r in over])) if over else 0.0  # noqa: E731
    agreement = []  # fraction of the exact (BF) accepted matches that FLANN also finds
    for p in overlapping:
        exact = {(g.queryIdx, g.trainIdx) for g in results[d, "BF"][p].good}
        found = {(g.queryIdx, g.trainIdx) for g in results[d, m][p].good}
        agreement.append(100 * len(exact & found) / max(len(exact), 1))
    return {"detector": d, "matcher": m, "tempo_ms": mean("tempo_ms"), "candidatos": mean("candidatos"),
            "aceitos": mean("aceitos"), "taxa_aceitacao_%": mean("taxa_aceitacao_%"), "inliers": mean("inliers"),
            "precisao_%": mean("precisao_%"), "concordancia_BF_%": float(np.mean(agreement)) if agreement else 0.0,
            "falsos_aceitos": float(np.mean([rows[d, m][p]["aceitos"] for p in separate])) if separate else 0.0}


def ratio_sweep(features: dict, results: dict, overlapping: list, matcher: str) -> list[dict]:
    """Re-filters the same kNN matches with several thresholds (requirement 3.2: choosing the threshold)."""
    rows = []
    for d in DETECTORS:
        for r in RATIOS:
            accepted, precision = [], []
            for i, j in overlapping:
                good, _ = ratio_test(results[d, matcher][i, j].knn, r)
                inliers = geometric_inliers(features[d][i], features[d][j], good).sum()
                accepted.append(len(good))
                precision.append(100 * inliers / max(len(good), 1))
            rows.append({"detector": d, "matcher": matcher, "ratio": r,
                         "aceitos": float(np.mean(accepted)) if accepted else 0.0,
                         "precisao_%": float(np.mean(precision)) if precision else 0.0})
    return rows


def pair_canvas(img_a: np.ndarray, img_b: np.ndarray, gap: int = 20) -> tuple[np.ndarray, int]:
    """Both images side by side (RGB) and the x offset of image B."""
    h = max(img_a.shape[0], img_b.shape[0])
    canvas = np.full((h, img_a.shape[1] + gap + img_b.shape[1], 3), 255, np.uint8)
    canvas[: img_a.shape[0], : img_a.shape[1]] = img_a
    canvas[: img_b.shape[0], img_a.shape[1] + gap:] = img_b
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), img_a.shape[1] + gap


def sample(matches: list, n: int = MAX_LINES) -> list:
    if len(matches) <= n:
        return matches
    return [matches[k] for k in np.random.default_rng(0).choice(len(matches), n, replace=False)]


def draw_lines(ax: plt.Axes, feat_a: Features, feat_b: Features, matches: list, offset: int, color: str,
               style: str = "-") -> None:
    if not matches:
        return
    pa = feat_a.points[[m.queryIdx for m in matches]]
    pb = feat_b.points[[m.trainIdx for m in matches]] + [offset, 0]
    ax.add_collection(LineCollection(np.stack([pa, pb], axis=1), colors=color, linestyles=style, linewidths=0.9))
    ax.scatter(np.r_[pa[:, 0], pb[:, 0]], np.r_[pa[:, 1], pb[:, 1]], s=6, c=color, edgecolors="black",
               linewidths=0.2, zorder=3)


def plot_before_after(image_set: ImageSet, i: int, j: int, feat_a: Features, feat_b: Features,
                      result: MatchResult, stats: dict, path: Path) -> None:
    """Requirement 3.3, as in Figure 3 of the assignment: accepted = solid, discarded = dashed."""
    canvas, offset = pair_canvas(image_set.images[i], image_set.images[j])
    h, w = canvas.shape[:2]
    fig, axes = plt.subplots(2, 1, figsize=(14, 2 * 14 * h / w + 1.6))
    candidates = sample(result.candidates)
    accepted, rejected = sample(result.good), sample(result.rejected, MAX_LINES // 3)

    axes[0].imshow(canvas)
    draw_lines(axes[0], feat_a, feat_b, candidates, offset, CANDIDATE_COLOR)
    axes[0].set_title(f"Antes do ratio test: {len(result.candidates)} candidatos (vizinho mais próximo), "
                      f"{len(candidates)} desenhados", fontsize=IMAGE_TITLE)
    axes[1].imshow(canvas)
    draw_lines(axes[1], feat_a, feat_b, rejected, offset, REJECTED_COLOR, "--")
    draw_lines(axes[1], feat_a, feat_b, accepted, offset, ACCEPTED_COLOR)
    axes[1].set_title(f"Depois do ratio test (limiar {result.ratio}): {stats['aceitos']} aceitos "
                      f"({stats['taxa_aceitacao_%']:.1f}%), {stats['inliers']} consistentes com uma homografia "
                      f"RANSAC (precisão {stats['precisao_%']:.1f}%)", fontsize=IMAGE_TITLE)
    axes[1].legend(handles=[Line2D([], [], color=ACCEPTED_COLOR, lw=2, label=f"aceito ({len(accepted)} desenhados)"),
                            Line2D([], [], color=REJECTED_COLOR, lw=2, ls="--",
                                   label=f"descartado ({len(rejected)} desenhados)")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.01), ncols=2, fontsize=IMAGE_TEXT, frameon=False)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{stats['par']}: {feat_a.method} + {result.method}, {stats['tempo_ms']:.1f} ms", fontsize=IMAGE_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))  # room for the suptitle
    save_figure(fig, path)


def plot_combinations(image_set: ImageSet, pair: tuple[int, int], features: dict, results: dict, rows: dict,
                      path: Path) -> None:
    """Accepted matches of every detector x matcher combination on the same pair."""
    i, j = pair
    canvas, offset = pair_canvas(image_set.images[i], image_set.images[j])
    h, w = canvas.shape[:2]
    fig, axes = plt.subplots(len(DETECTORS), len(MATCHERS), figsize=(10 * len(MATCHERS), len(DETECTORS) * 10 * h / w + 1.8))
    for (r, d), (c, m) in itertools.product(enumerate(DETECTORS), enumerate(MATCHERS)):
        ax, stats = axes[r, c], rows[d, m][pair]
        ax.imshow(canvas)
        draw_lines(ax, features[d][i], features[d][j], sample(results[d, m][pair].good), offset, ACCEPTED_COLOR)
        ax.set_title(f"{d} + {m}: {stats['aceitos']} aceitos, {stats['inliers']} inliers "
                     f"(precisão {stats['precisao_%']:.1f}%), {stats['tempo_ms']:.1f} ms", fontsize=IMAGE_TITLE)
        ax.axis("off")
    fig.suptitle(f"{rows['SIFT', 'BF'][pair]['par']}: matches aceitos pelo ratio test em cada combinação "
                 f"(até {MAX_LINES} desenhados)", fontsize=IMAGE_SUPTITLE)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))  # room for the suptitle
    save_figure(fig, path)


def plot_metrics(summary: list[dict], n_pairs: int, path: Path) -> None:
    """Small multiples; bar color = detector, hatched bar = FLANN."""
    panels = [("tempo_ms", "Tempo de emparelhamento (ms)", "{:.1f}"),
              ("aceitos", "Matches aceitos pelo ratio test", "{:.0f}"),
              ("inliers", "Inliers RANSAC", "{:.0f}"),
              ("precisao_%", "Precisão: inliers / aceitos (%)", "{:.1f}"),
              ("concordancia_BF_%", "Concordância com o BF (%)", "{:.1f}"),
              ("falsos_aceitos", "Aceitos em pares sem sobreposição", "{:.1f}")]
    labels = [f"{r['detector']}\n{r['matcher']}" for r in summary]
    colors = [DETECTOR_COLORS[r["detector"]] for r in summary]
    hatches = ["//" if r["matcher"] == "FLANN" else "" for r in summary]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (key, title, fmt) in zip(axes.flat, panels):
        values = [r[key] for r in summary]
        ax.bar(labels, values, color=colors, hatch=hatches, edgecolor="white", width=0.7)
        top = max(values) if max(values) > 0 else 1
        for x, v in enumerate(values):
            ax.text(x, v + 0.02 * top, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)
        ax.set_ylim(0, top * 1.15)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle(f"Comparação detector x matcher: média sobre {n_pairs} pares com sobreposição "
                 f"(barras hachuradas = FLANN)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))  # room for the suptitle
    save_figure(fig, path)


def plot_ratio_sweep(sweep: list[dict], matcher: str, ratio: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, title in ((axes[0], "aceitos", "Matches aceitos"), (axes[1], "precisao_%", "Precisão: inliers / aceitos (%)")):
        for d in DETECTORS:
            rows = [r for r in sweep if r["detector"] == d]
            ax.plot([r["ratio"] for r in rows], [r[key] for r in rows], color=DETECTOR_COLORS[d], lw=2, marker="o",
                    ms=6, label=d)
        ax.axvline(ratio, color=INK_SECONDARY, ls="--", lw=1)
        ax.text(ratio, ax.get_ylim()[1], f" limiar usado: {ratio}", va="top", fontsize=9, color=INK_SECONDARY)
        ax.set_xlabel("limiar do ratio test")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    axes[0].legend(fontsize=10)
    fig.suptitle(f"Curva do ratio test ({matcher}, média sobre os pares com sobreposição)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))  # room for the suptitle
    save_figure(fig, path)
