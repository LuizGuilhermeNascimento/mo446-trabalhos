"""Automatic image ordering (assignment: Etapa 4, requirements 4.1-4.5).

1. Every pair is matched (step 2) and verified with a RANSAC homography (step 4): the number of
   inliers of each pair forms the connectivity matrix.
2. A pair overlaps when n_inliers >= max(MIN_INLIERS, ALPHA + BETA * n_matches) (Brown & Lowe, 2007):
   this is the adjacency matrix of the neighborhood graph.
3. The largest connected component is the panorama; images outside it are rejected intruders.
4. The maximum spanning tree links each image to its most reliable neighbor; its longest path
   (diameter) is the inferred sequence, oriented left -> right (or top -> bottom).
5. The middle image of the sequence is the reference of the panorama (least projective distortion).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

from common import (CRITICAL, IMAGE_SUPTITLE, IMAGE_TEXT, INK, INK_MUTED, INK_SECONDARY, ImageSet, log,
                    print_table, save_figure, write_csv)
from step1_key_points_detection.detector import Features
from step2_feature_matching.matcher import FeatureMatcher
from step4_homography_connection.homography import PairHomography, estimate_pair_homography, plot_ransac

STAGE = Path(__file__).resolve().parent.name
MIN_INLIERS = 20  # default heuristic: minimum inliers for a pair to overlap
ALPHA, BETA = 8.0, 0.3  # Brown & Lowe (2007) verification: n_inliers > ALPHA + BETA * n_matches

HEATMAP = LinearSegmentedColormap.from_list("inliers", ["#f7f7f5", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
TREE_COLOR = "#256abf"  # nodes and edges of the inferred sequence (maximum spanning tree)


@dataclass
class SortingResult:
    names: list[str]
    match_matrix: np.ndarray  # N x N ratio-test matches per pair
    inlier_matrix: np.ndarray  # N x N RANSAC inliers per pair: the connectivity matrix
    required: np.ndarray  # N x N inliers needed for the pair to overlap
    adjacency: np.ndarray  # N x N bool: pairs that overlap
    pairs: dict[tuple[int, int], PairHomography]  # every pair (i < j)
    accepted: list[int]  # images of the panorama
    rejected: list[int]  # intruders
    tree_edges: list[tuple[int, int]]  # maximum spanning tree (used by pairwise chaining)
    overlap_edges: list[tuple[int, int]]  # every accepted overlap (used by bundle adjustment)
    order: list[int]  # inferred sequence
    reference: int  # reference image of the panorama

    def homography(self, i: int, j: int) -> PairHomography:
        """Pair estimate oriented from i to j."""
        return self.pairs[i, j] if i < j else self.pairs[j, i].inverse()


class ImageSorter:
    def __init__(self, matcher: FeatureMatcher, min_inliers: int = MIN_INLIERS, alpha: float = ALPHA,
                 beta: float = BETA):
        self.matcher = matcher
        self.min_inliers, self.alpha, self.beta = min_inliers, alpha, beta

    def sort(self, image_set: ImageSet, features: list[Features]) -> SortingResult:
        names, n = image_set.names, len(features)
        pairs = {}
        for i, j in itertools.combinations(range(n), 2):
            matches = self.matcher.match(features[i], features[j]).good
            pairs[i, j] = estimate_pair_homography(i, j, features[i], features[j], matches)

        matches, inliers, required = (np.zeros((n, n), int) for _ in range(3))
        for (i, j), pair in pairs.items():
            matches[i, j] = matches[j, i] = pair.n_matches
            inliers[i, j] = inliers[j, i] = pair.n_inliers
            required[i, j] = required[j, i] = max(self.min_inliers, int(np.ceil(self.alpha + self.beta * pair.n_matches)))
        adjacency = (inliers >= required) & ~np.eye(n, dtype=bool)

        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_weighted_edges_from((i, j, inliers[i, j]) for i, j in pairs if adjacency[i, j])
        main = max(nx.connected_components(graph), key=lambda c: (len(c), graph.subgraph(c).size("weight")))
        if len(main) < 2:
            raise RuntimeError("Nenhum par de imagens se sobrepõe: não é possível montar o panorama.")
        tree = nx.maximum_spanning_tree(graph.subgraph(main), weight="weight")

        order = longest_path(tree)
        if len(order) < len(main):  # branched tree: the remaining images follow in BFS order
            log("Passo 3", "aviso: a árvore de vizinhança não é linear; imagens fora do caminho vão ao final")
            order += [k for k in nx.bfs_tree(tree, order[0]) if k not in order]
        order = orient(order, pairs, [image.shape for image in image_set.images])

        result = SortingResult(
            names=names, match_matrix=matches, inlier_matrix=inliers, required=required, adjacency=adjacency,
            pairs=pairs, accepted=sorted(main), rejected=[k for k in range(n) if k not in main],
            tree_edges=[tuple(sorted(e)) for e in tree.edges],
            overlap_edges=[(i, j) for i, j in pairs if adjacency[i, j] and i in main],
            order=order, reference=order[len(order) // 2])
        log("Passo 3", f"sequência inferida: {' -> '.join(names[k] for k in order)}")
        log("Passo 3", f"referência: {names[result.reference]}; intrusas rejeitadas: "
                       f"{', '.join(names[k] for k in result.rejected) or 'nenhuma'}")
        return result


def longest_path(tree: nx.Graph) -> list[int]:
    """Diameter of the tree: farthest node u from any node, then the farthest node from u."""
    start = min(tree.nodes)
    u = max(nx.single_source_shortest_path_length(tree, start).items(), key=lambda kv: kv[1])[0]
    paths = nx.single_source_shortest_path(tree, u)
    return paths[max(paths, key=lambda v: len(paths[v]))]


def orient(order: list[int], pairs: dict, shapes: list[tuple]) -> list[int]:
    """Left -> right (or top -> bottom): where does the center of image order[1] fall in the frame of order[0]?"""
    a, b = order[0], order[1]
    pair = pairs[a, b] if a < b else pairs[b, a].inverse()  # maps a -> b
    center = lambda k: np.float64([shapes[k][1] / 2, shapes[k][0] / 2])  # noqa: E731
    in_a = cv2.perspectiveTransform(center(b).reshape(1, 1, 2), np.linalg.inv(pair.H)).ravel()
    dx, dy = in_a - center(a)
    forward = dx > 0 if abs(dx) >= abs(dy) else dy > 0
    return order if forward else order[::-1]


def save_sorting_outputs(image_set: ImageSet, result: SortingResult, out_dir: Path) -> None:
    names, n = result.names, len(result.names)
    display = result.order + result.rejected  # rows/columns of the matrices in the inferred order

    decision = []
    for k in display:
        others = [m for m in range(n) if m != k]
        best = max(others, key=lambda m: result.inlier_matrix[k, m])
        decision.append({"imagem": names[k],
                         "posicao": result.order.index(k) + 1 if k in result.order else "-",
                         "melhor_par": names[best], "inliers_melhor_par": result.inlier_matrix[k, best],
                         "limiar_melhor_par": result.required[k, best],
                         "vizinhos_sobrepostos": int(result.adjacency[k].sum()),
                         "situacao": "intrusa (rejeitada)" if k in result.rejected
                         else "referência" if k == result.reference else "aceita"})
    matrix_rows = [{"imagem": names[r], **{names[c]: result.inlier_matrix[r, c] for c in display}} for r in display]
    pair_rows = [{"par": f"{names[i]} x {names[j]}", "matches": p.n_matches, "inliers": p.n_inliers,
                  "taxa_inliers_%": 100 * p.inlier_ratio, "limiar": result.required[i, j],
                  "sobrepoe": "sim" if result.adjacency[i, j] else "não",
                  "na_arvore": "sim" if (i, j) in result.tree_edges else "não"} for (i, j), p in result.pairs.items()]

    print_table(matrix_rows, "Passo 3: matriz de conectividade (inliers RANSAC por par, na ordem inferida)")
    print_table(decision, f"Passo 3: decisão por imagem (o par se sobrepõe se inliers >= limiar = "
                          f"max(inliers mínimos, {ALPHA:.0f} + {BETA} x matches))")
    print(f"Sequência inferida: {' -> '.join(names[k] for k in result.order)}\n", flush=True)

    write_csv(matrix_rows, out_dir / "matriz_conectividade.csv")
    write_csv([{"imagem": names[r], **{names[c]: int(result.adjacency[r, c]) for c in display}} for r in display],
              out_dir / "matriz_adjacencia.csv")
    write_csv(pair_rows, out_dir / "pares.csv")
    write_csv(decision, out_dir / "decisao_por_imagem.csv")

    group = image_set.group
    plot_matrix(result, display, group, out_dir / "matriz_conectividade.png")
    plot_graph(result, group, out_dir / "grafo_vizinhanca.png")
    plot_sequence(image_set, result, out_dir / "sequencia_inferida.jpg")
    for k in result.rejected:  # evidence of the rejection: RANSAC on the intruder's best pair
        best = best_partner(result, k)
        pair = result.homography(k, best)
        found = ("Nenhuma Correspondência Consistente" if pair.n_inliers == 0 else
                 f"Apenas {pair.n_inliers} Correspondências Consistentes (Mínimo {result.required[k, best]})")
        plot_ransac(image_set.images[k], image_set.images[best], pair,
                    f"Imagem Intrusa {names[k]} e Melhor Candidata {names[best]}: {found}",
                    out_dir / f"intrusa_{names[k]}.jpg")
    log("Passo 3", f"saídas salvas em {out_dir}")


def best_partner(result: SortingResult, k: int) -> int:
    return max((m for m in range(len(result.names)) if m != k), key=lambda m: result.inlier_matrix[k, m])


def plot_matrix(result: SortingResult, display: list[int], group: str, path: Path) -> None:
    """Heatmap of the connectivity matrix in the inferred order; outlined cells = pairs that overlap
    (as in Figure 4 of the assignment); intruders labeled in red."""
    m = result.inlier_matrix[np.ix_(display, display)]
    labels = [result.names[k] for k in display]
    size = max(6.0, 0.85 * len(display) + 2)
    fig, ax = plt.subplots(figsize=(size + 1.5, size + 1.2), layout="constrained")
    image = ax.imshow(m, cmap=HEATMAP)
    for r, c in itertools.product(range(len(display)), repeat=2):
        if r == c:
            ax.text(c, r, "—", ha="center", va="center", color=INK_MUTED, fontsize=11)
            continue
        dark = m[r, c] > 0.55 * m.max()
        ax.text(c, r, str(m[r, c]), ha="center", va="center", fontsize=11, color="white" if dark else INK)
        if result.adjacency[display[r], display[c]]:  # clip_on=False: border cells keep their whole outline
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.2, clip_on=False))
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=11)
    ax.set_yticks(range(len(labels)), labels, fontsize=11)
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        if result.names.index(tick.get_text()) in result.rejected:
            tick.set_color(CRITICAL)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="inliers RANSAC")
    fig.legend(handles=[Patch(facecolor="none", edgecolor=INK, lw=1.2, label="par com sobreposição"),
                        Line2D([], [], color=CRITICAL, lw=0, marker="s", label="imagem intrusa (rótulo em vermelho)")],
               loc="outside lower center", ncols=2, fontsize=10, frameon=False)
    fig.suptitle(f"Matriz de Conectividade (Inliers RANSAC por Par): Conjunto {group}", fontsize=13)
    save_figure(fig, path)


def plot_graph(result: SortingResult, group: str, path: Path) -> None:
    """Neighborhood graph (Figure 4 of the assignment): the inferred sequence on a line, each edge of the
    maximum spanning tree labeled with its inliers; each intruder right below its best candidate, linked by
    a dashed red line crossed out and labeled with its (too few) inliers."""
    names, order = result.names, result.order
    pos = {k: (float(n), 0.0) for n, k in enumerate(order)}
    below: dict[int, int] = {}
    for k in result.rejected:  # below the best candidate (side by side when several share it)
        best = best_partner(result, k)
        x = pos[best][0] if best in pos else (len(order) - 1) / 2
        pos[k] = (x + 0.9 * below.get(best, 0), -1.3)
        below[best] = below.get(best, 0) + 1

    fig, ax = plt.subplots(figsize=(1.4 * len(order) + 1.0, 3.8), layout="constrained")
    for i, j in result.tree_edges:
        (x1, y1), (x2, y2) = pos[i], pos[j]
        adjacent = abs(order.index(i) - order.index(j)) == 1
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-", color=TREE_COLOR, lw=3, zorder=1,
                                     connectionstyle="arc3" if adjacent else "arc3,rad=-0.35",
                                     shrinkA=0, shrinkB=0))
        ax.text((x1 + x2) / 2, 0.12 if adjacent else 0.5, str(result.inlier_matrix[i, j]), ha="center",
                va="bottom", fontsize=11, color=INK)
    for k in result.rejected:
        best = best_partner(result, k)
        (x1, y1), (x2, y2) = pos[k], pos[best]
        y1 += 0.22  # from the top of the intruder node...
        y2 -= 0.85 if best == result.reference else 0.62  # ...to just below the candidate's name
        ax.plot([x1, x2], [y1, y2], color=CRITICAL, lw=2, ls="--", zorder=1)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        ax.scatter(xm, ym, s=160, marker="x", color=CRITICAL, linewidths=3, zorder=2)
        ax.text(xm + 0.12, ym, str(result.inlier_matrix[k, best]), ha="left", va="center", fontsize=11,
                color=CRITICAL)
    for k, (x, y) in pos.items():
        rejected, reference = k in result.rejected, k == result.reference
        color = CRITICAL if rejected else INK if reference else TREE_COLOR
        ax.scatter(x, y, s=1100, color=color, edgecolors="white", linewidths=2, zorder=3)
        ax.text(x, y, "X" if rejected else str(order.index(k) + 1), ha="center", va="center", fontsize=13,
                color="white", zorder=4)
        extra = "\nintrusa rejeitada" if rejected else "\nreferência" if reference else ""
        ax.text(x, y - 0.3, names[k] + extra, ha="center", va="top", fontsize=10,
                color=CRITICAL if rejected else INK_SECONDARY)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(-2.0, 0.9 if any(abs(order.index(i) - order.index(j)) > 1 for i, j in result.tree_edges) else 0.5)
    ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=TREE_COLOR, lw=3, label="par vizinho (número = inliers)"),
                        Line2D([], [], color=CRITICAL, lw=2, ls="--", marker="x", ms=9, mew=2.5,
                               label="intrusa rejeitada")],
               loc="outside lower center", ncols=2, fontsize=11, frameon=False)
    fig.suptitle(f"Grafo de Vizinhança e Sequência Inferida: Conjunto {group}", fontsize=14)
    save_figure(fig, path, dpi=220)


def plot_sequence(image_set: ImageSet, result: SortingResult, path: Path) -> None:
    """Thumbnails in the inferred order, intruders at the end crossed out."""
    shown = result.order + result.rejected
    h, w = image_set.images[result.order[0]].shape[:2]
    panel = 3.3
    fig, axes = plt.subplots(1, len(shown), figsize=(panel * len(shown), panel * h / w + 1.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, wspace=0.005)  # images close together
    for ax, k in zip(np.atleast_1d(axes), shown):
        ax.imshow(cv2.cvtColor(image_set.images[k], cv2.COLOR_BGR2RGB))
        if k in result.rejected:
            ih, iw = image_set.images[k].shape[:2]
            ax.plot([0, iw - 1], [0, ih - 1], color=CRITICAL, lw=3)
            ax.plot([0, iw - 1], [ih - 1, 0], color=CRITICAL, lw=3)
            ax.set_xlim(-0.5, iw - 0.5)  # same extent as the other panels
            ax.set_ylim(ih - 0.5, -0.5)
            title, color = f"intrusa\n{result.names[k]}", CRITICAL
        else:
            ref = " (referência)" if k == result.reference else ""
            title, color = f"{result.order.index(k) + 1}{ref}\n{result.names[k]}", INK
        ax.set_title(title, fontsize=IMAGE_TEXT, color=color)
        ax.axis("off")
    fig.suptitle(f"Sequência Inferida: Conjunto {image_set.group}", fontsize=IMAGE_SUPTITLE)
    save_figure(fig, path)
