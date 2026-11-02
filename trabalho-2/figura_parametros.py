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
GAPS = {1: "vizinhas", 2: "dois passos"}
FAIL_DEG = 5.0
NO_POSE = 400.0


def complete(rows, **fixed):
    # Keep pairs that produced a pose at every ratio, so each median is over the same pairs.
    sel = [r for r in rows if all(r[k] == v for k, v in fixed.items())]
    ok = {p for p in {r["par"] for r in sel} if {r["ratio"] for r in sel if r["par"] == p} >= set(RATIOS)}
    return [r for r in sel if r["par"] in ok]


def failure_pct(rows, detector, gap):
    # Configurations that produced no pose at all count as failures.
    grid = {(features.MAX_SIDE, r, features.F_THRESHOLD_PX) for r in RATIOS}
    grid |= {(features.MAX_SIDE, features.RATIO, t) for t in THRESHOLDS_PX}
    grid |= {(s, features.RATIO, features.F_THRESHOLD_PX) for s in SIDES}
    pairs = {r["par"] for r in rows if r["gap"] == gap}
    done = {(r["par"], r["lado_px"], r["ratio"], r["limiar_px"]): r for r in rows if r["detector"] == detector and r["gap"] == gap}
    fails = [(p, *g) not in done or done[(p, *g)]["erro_R_graus"] > FAIL_DEG for p in pairs for g in grid]
    return 100 * np.mean(fails)


def default_pairs(rows, detector, gap):
    # Pose errors per pair at the default configuration; None when no pose was estimated.
    done = {r["par"]: r for r in rows if r["detector"] == detector and r["gap"] == gap
            and (r["lado_px"], r["ratio"], r["limiar_px"]) == (features.MAX_SIDE, features.RATIO, features.F_THRESHOLD_PX)}
    return [done.get(p) for p in sorted({r["par"] for r in rows if r["gap"] == gap})]


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
            xs, ys = median_curve(complete(rows[o], detector="SIFT", gap=1, **base), "ratio", y)
            ax.plot(xs, ys, "o-", color=f"C{k}", label=label)
    style(axes[0], "limiar do ratio test", "correspondências aceitas/par")
    style(axes[1], "limiar do ratio test", "inliers/aceitas (%)")
    axes[1].legend(fontsize=8)
    for name, (color, marker) in DETECTORS.items():
        for gap, ls in ((1, "-"), (2, "--")):
            pooled = [r for o in OBJECTS for r in complete(rows[o], detector=name, gap=gap, **base)]
            xs, ys = median_curve(pooled, "ratio", "inliers_F_pct")
            axes[2].plot(xs, ys, ls, marker=marker, color=color, label=name if gap == 1 else None)
    for gap, ls in ((1, "-"), (2, "--")):
        axes[2].plot([], [], ls, color="0.4", label=GAPS[gap])
    style(axes[2], "limiar do ratio test", "inliers/aceitas (%)")
    axes[2].legend(fontsize=7, ncol=2, loc="lower left")
    for ax in axes:
        ax.axvline(features.RATIO, color="0.3", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(saida / "hiperparametros_objetos.png", dpi=200)


def pose_figure(rows, saida):
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True, sharey=True)
    for col, (o, label) in enumerate(OBJECTS.items()):
        for row, (key, ylabel) in enumerate((("erro_R_graus", "erro de rotação (°)"), ("erro_t_graus", "erro da direção de t (°)"))):
            ax = axes[row, col]
            for j, (name, (color, marker)) in enumerate(DETECTORS.items()):
                for g, gap in enumerate(GAPS):
                    pairs = default_pairs(rows[o], name, gap)
                    x = g + (j - 1) * 0.25 + np.linspace(-0.06, 0.06, len(pairs))
                    ys = [max(r[key], 0.01) if r else NO_POSE for r in pairs]
                    ax.scatter(x, ys, s=22, marker=marker, color=color, edgecolor="white", lw=0.5, zorder=3,
                               label=name if g == 0 else None)
                    found = [y for y in ys if y < NO_POSE]
                    ax.hlines(np.median(found), x[0] - 0.05, x[-1] + 0.05, color="0.15", lw=1.5, zorder=4)
            ax.axhline(FAIL_DEG, color="0.3", ls=":", lw=1)
            ax.axhspan(200, 1000, color="0.92", zorder=0)
            ax.set_yscale("log")
            ax.set_ylim(0.006, 1000)
            ax.set_xticks(range(len(GAPS)), list(GAPS.values()))
            style(ax, "", ylabel if col == 0 else "")
            if row == 0:
                ax.set_title(label)
    axes[0, 0].text(-0.45, NO_POSE, "sem pose", va="center", fontsize=7, color="0.3")
    axes[0, 2].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(saida / "detectores_pose.png", dpi=200)


def table(rows, saida):
    # Per gap: median inliers, inlier %, rotation and translation errors, and failure % over the sweep.
    digits = [0, 0, 1, 1, 0] * len(GAPS)
    better = [max, max, min, min, min] * len(GAPS)
    lines = [r"\begin{tabular}{ll" + "rrrrr" * len(GAPS) + "}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{5}}{{c}}{{{g.capitalize()}}}" for g in GAPS.values()) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 5 * k}-{7 + 5 * k}}}" for k in range(len(GAPS))),
             "Objeto & Detector" + r" & Inl. & Inl.\,(\%) & $e_R$ & $e_t$ & Falha\,(\%)" * len(GAPS) + r" \\"]
    for o, label in OBJECTS.items():
        block = {}
        for name in DETECTORS:
            cells = []
            for gap in GAPS:
                found = [r for r in default_pairs(rows[o], name, gap) if r]
                cells += [np.median([r[k] for r in found]) for k in ("inliers_F", "inliers_F_pct", "erro_R_graus", "erro_t_graus")]
                cells.append(failure_pct(rows[o], name, gap))
            block[name] = [round(v, d) for v, d in zip(cells, digits)]
        # Best detector per object and column in bold.
        best = [f(c[i] for c in block.values()) for i, f in enumerate(better)]
        lines.append(r"\midrule")
        for k, (name, cells) in enumerate(block.items()):
            text = [f"{v:.{d}f}".replace(".", ",") for v, d in zip(cells, digits)]
            text = [rf"\textbf{{{t}}}" if v == b else t for t, v, b in zip(text, cells, best)]
            lines.append((label if k == 0 else "") + f" & {name} & " + " & ".join(text) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (saida / "detectores_objetos.tex").write_text("\n".join(lines) + "\n")


def main(saida):
    rows = {o: json.loads((saida / f"{o}_parametros/parametros.json").read_text())["dois_pontos"] for o in OBJECTS}
    ratio_figure(rows, saida)
    pose_figure(rows, saida)
    table(rows, saida)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
