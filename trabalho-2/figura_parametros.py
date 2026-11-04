import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import features
from parametros import RATIOS, SIDES, THRESHOLDS_PX, median_curve

OBJECTS = {"moinho": "Moinho", "pato": "Pato", "pedra": "Pedra"}
DETECTORS = {"SIFT": ("#d62728", "o"), "ORB": ("#7b4fb3", "s"), "AKAZE": ("#009a9a", "^")}
FAIL_DEG = 5.0


def complete(rows, **fixed):
    # Keep pairs that produced a pose at every ratio, so each median is over the same pairs.
    sel = [r for r in rows if all(r[k] == v for k, v in fixed.items())]
    ok = {p for p in {r["par"] for r in sel} if {r["ratio"] for r in sel if r["par"] == p} >= set(RATIOS)}
    return [r for r in sel if r["par"] in ok]


def failure_pct(rows, detector):
    # Configurations that produced no pose at all count as failures.
    grid = {(features.MAX_SIDE, r, features.F_THRESHOLD_PX) for r in RATIOS}
    grid |= {(features.MAX_SIDE, features.RATIO, t) for t in THRESHOLDS_PX}
    grid |= {(s, features.RATIO, features.F_THRESHOLD_PX) for s in SIDES}
    pairs = {r["par"] for r in rows}
    done = {(r["par"], r["lado_px"], r["ratio"], r["limiar_px"]): r for r in rows if r["detector"] == detector}
    fails = [(p, *g) not in done or done[(p, *g)]["erro_R_graus"] > FAIL_DEG for p in pairs for g in grid]
    return 100 * np.mean(fails)


def default_pairs(rows, detector):
    # Pose errors per pair at the default configuration; None when no pose was estimated.
    done = {r["par"]: r for r in rows if r["detector"] == detector
            and (r["lado_px"], r["ratio"], r["limiar_px"]) == (features.MAX_SIDE, features.RATIO, features.F_THRESHOLD_PX)}
    return [done.get(p) for p in sorted({r["par"] for r in rows})]


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def ratio_figure(rows, saida):
    base = {"lado_px": features.MAX_SIDE, "limiar_px": features.F_THRESHOLD_PX}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for k, (o, label) in enumerate(OBJECTS.items()):
        for ax, y in zip(axes[:2], ("matches", "inliers_F_pct")):
            xs, ys = median_curve(complete(rows[o], detector="SIFT", **base), "ratio", y)
            ax.plot(xs, ys, "o-", color=f"C{k}", label=label)
    style(axes[0], "limiar do ratio test", "correspondências aceitas/par")
    style(axes[1], "limiar do ratio test", "inliers/aceitas (%)")
    axes[1].legend(fontsize=8)
    for name, (color, marker) in DETECTORS.items():
        pooled = [r for o in OBJECTS for r in complete(rows[o], detector=name, **base)]
        xs, ys = median_curve(pooled, "ratio", "inliers_F_pct")
        axes[2].plot(xs, ys, "-", marker=marker, color=color, label=name)
    style(axes[2], "limiar do ratio test", "inliers/aceitas (%)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvline(features.RATIO, color="0.3", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(saida / "hiperparametros_objetos.png", dpi=200)


def table(rows, saida):
    # Median inliers, inlier %, rotation and translation errors, and failure % over the sweep.
    digits, better = [0, 0, 1, 1, 0], [max, max, min, min, min]
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Objeto & Detector & \textit{Inliers} & \textit{Inliers} (\%) & Erro $R$ & Erro $t$ & Falha (\%) \\"]
    for o, label in OBJECTS.items():
        block = {}
        for name in DETECTORS:
            found = [r for r in default_pairs(rows[o], name) if r]
            cells = [np.median([r[k] for r in found]) for k in ("inliers_F", "inliers_F_pct", "erro_R_graus", "erro_t_graus")]
            block[name] = [round(v, d) for v, d in zip(cells + [failure_pct(rows[o], name)], digits)]
        # Best detector per object and column in bold.
        best = [f(c[i] for c in block.values()) for i, f in enumerate(better)]
        lines.append(r"\midrule")
        for k, (name, cells) in enumerate(block.items()):
            text = [f"{v:.{d}f}".replace(".", ",") for v, d in zip(cells, digits)]
            text = [rf"\textbf{{{t}}}" if v == b else t for t, v, b in zip(text, cells, best)]
            text = text[:2] + [t + r"$^\circ$" for t in text[2:4]] + text[4:]
            lines.append((label if k == 0 else "") + f" & {name} & " + " & ".join(text) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (saida / "detectores_objetos.tex").write_text("\n".join(lines) + "\n")


def main(saida):
    rows = {o: json.loads((saida / f"{o}_parametros/parametros.json").read_text())["dois_pontos"] for o in OBJECTS}
    ratio_figure(rows, saida)
    table(rows, saida)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
