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
DETECTORS = {"SIFT": "C3", "ORB": "C4", "AKAZE": "C5"}
FAIL_DEG = 5.0


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


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def main(saida):
    rows = {o: json.loads((saida / f"{o}_parametros/parametros.json").read_text())["dois_pontos"] for o in OBJECTS}
    base = {"lado_px": features.MAX_SIDE, "limiar_px": features.F_THRESHOLD_PX}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))

    for k, (o, label) in enumerate(OBJECTS.items()):
        for ax, y in zip(axes[:2], ("matches", "inliers_F_pct")):
            xs, ys = median_curve(complete(rows[o], detector="SIFT", gap=1, **base), "ratio", y)
            ax.plot(xs, ys, "o-", color=f"C{k}", label=label)
    style(axes[0], "limiar do ratio test", "correspondências aceitas/par")
    style(axes[1], "limiar do ratio test", "inliers/aceitas (%)")
    axes[1].legend(fontsize=8)

    for name, color in DETECTORS.items():
        for gap, ls in ((1, "o-"), (2, "s--")):
            pooled = [r for o in OBJECTS for r in complete(rows[o], detector=name, gap=gap, **base)]
            xs, ys = median_curve(pooled, "ratio", "inliers_F_pct")
            axes[2].plot(xs, ys, ls, color=color, label=f"{name}, {gap} passo{'s' * (gap > 1)}")
    style(axes[2], "limiar do ratio test", "inliers/aceitas (%)")
    axes[2].legend(fontsize=7, ncol=1)
    for ax in axes[:3]:
        ax.axvline(features.RATIO, color="0.3", ls=":", lw=1)

    width = 0.25
    for j, (name, color) in enumerate(DETECTORS.items()):
        fails = [failure_pct(rows[o], name, 2) for o in OBJECTS]
        axes[3].bar(np.arange(len(OBJECTS)) + (j - 1) * width, fails, width, color=color, label=name)
    axes[3].set_xticks(range(len(OBJECTS)), list(OBJECTS.values()))
    style(axes[3], "", f"pose ausente ou com erro > {FAIL_DEG:g}° (%)")
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(saida / "hiperparametros_objetos.png", dpi=200)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
