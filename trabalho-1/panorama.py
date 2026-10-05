"""Panorama pipeline: from a folder of out-of-order images to the final panorama.

Usage (from trabalho-1/):
    python panorama.py input/<grupo>                 # every step, panorama saved to output/final/<grupo>/
    python panorama.py input/<grupo> --ate-passo 3   # stop after a step (1 to 5)
    python panorama.py --todos                       # every group folder inside input/
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common import (INPUT_DIR, OUTPUT_DIR, list_image_files, load_image_set, log, print_table, save_image, set_seed,
                    stage_output_dir)
from step1_key_points_detection import detector as step1
from step2_feature_matching import matcher as step2
from step3_image_sorting import sorter as step3
from step4_homography_connection import alignment as step4
from step5_panorama_composition import compositor as step5
from step5_panorama_composition import evaluation as step5_evaluation

STEPS = {
    1: "Detecção de características", 
    2: "Emparelhamento", 
    3: "Ordenação automática",
    4: "Homografia e alinhamento", 
    5: "Composição e remoção de fantasmas"
}


@dataclass
class Config:
    max_side: int = 1024  # largest image side after resizing (px)
    shuffle_seed: int | None = 0  # input order shuffled with this seed (requirement 4.1); None = alphabetical
    max_features: int = 3000  # keypoints kept per image (step 1)
    detector: str = "SIFT"  # SIFT, ORB or AKAZE (step 1 study)
    ratio: float = step2.RATIO  # Lowe's ratio test threshold (step 2 ratio curve)
    matching_pair: tuple[str, str] | None = ("20260926_3", "20260926_4")  # pair seen by every detector in step 2
    #   (it shows the paraglider); groups without these images fall back to the pair with the median inliers
    min_inliers: int = step3.MIN_INLIERS  # minimum inliers for two images to overlap (step 3)
    projection: str = "cylindrical"  # cylindrical or planar (step 4; the groups span ~170-220 degrees)
    alignment: str = "pairwise"  # pairwise or bundle (step 4 study)
    deghost: str = "seam"  # seam or none (step 5 study)
    blend: str = "feather"  # feather, multiband or none (step 5 study)
    # Regions (x, y, w, h in the final cropped panorama) enlarged in step 5, per group; groups not listed
    # get the region with the strongest ghosts. "fantasma": moving object; "costura": seam through a nearby object.
    zoom_regions: dict = field(default_factory=lambda: {
        "paisagem": {"fantasma": (1110, 215, 160, 160), "costura": (620, 0, 260, 180)}})


CONFIG = Config()


def run(folder: str | Path, until: int = 5, config: Config = CONFIG) -> Path | None:
    """Runs steps 1..`until` on one group folder; returns the panorama path when step 5 runs."""
    set_seed()
    times: dict[int, float] = {}
    start = time.perf_counter()
    image_set = load_image_set(folder, config.max_side, config.shuffle_seed)
    group = image_set.group
    print_capture_metadata(Path(folder), group)
    print_table([{"parâmetro": k, "valor": v} for k, v in asdict(config).items()], f"Parâmetros do pipeline ({group})")

    # Step 1: every detector is studied; the chosen one feeds the next steps.
    log("Pipeline", f"Passo 1: {STEPS[1]}")
    features = step1.compare_detectors(image_set, stage_output_dir(step1.STAGE, group), config.max_features)
    times[1] = time.perf_counter() - start
    if until == 1:
        return finish(group, times)

    # Step 3 runs before the step 2 outputs: the matching study needs its graph (overlaps and neighbors).
    start = time.perf_counter()
    sorting = step3.ImageSorter(step2.FeatureMatcher(config.ratio), config.min_inliers).sort(
        image_set, features[config.detector])
    times[3] = time.perf_counter() - start

    start = time.perf_counter()
    log("Pipeline", f"passo 2: {STEPS[2]}")
    step2.compare_matching(image_set, stage_output_dir(step2.STAGE, group), features, sorting.overlap_edges,
                           sorting.order, config.ratio, config.detector, config.matching_pair)
    times[2] = time.perf_counter() - start
    if until == 2:
        times[2] += times.pop(3)  # the sorting only ran to support the step 2 study
        return finish(group, times)

    start = time.perf_counter()
    log("Pipeline", f"passo 3: {STEPS[3]}")
    step3.save_sorting_outputs(image_set, sorting, stage_output_dir(step3.STAGE, group))
    times[3] += time.perf_counter() - start
    if until == 3:
        return finish(group, times)

    start = time.perf_counter()
    log("Pipeline", f"passo 4: {STEPS[4]}")
    alignment = step4.align(image_set.images, sorting.pairs, sorting.tree_edges, sorting.overlap_edges,
                            sorting.reference, config.alignment, config.projection, compare_modes=True)
    step4.save_alignment_outputs(image_set.names, alignment, sorting.order, sorting.overlap_edges, group,
                                 stage_output_dir(step4.STAGE, group))
    times[4] = time.perf_counter() - start
    if until == 4:
        return finish(group, times)

    start = time.perf_counter()
    log("Pipeline", f"passo 5: {STEPS[5]}")
    out_dir = stage_output_dir(step5_evaluation.STAGE, group)
    compositor = step5.PanoramaCompositor(config.deghost, config.blend)
    result = compositor.compose(alignment.aligned)
    regions = config.zoom_regions.get(group, {})
    step5_evaluation.save_composition_outputs(image_set.names, alignment, compositor.name, result, group, out_dir,
                                              regions.get("fantasma"))
    step5_evaluation.compare_compositions(alignment, regions.get("costura"), group, out_dir)
    panorama_path = OUTPUT_DIR / "final" / group / "panorama.jpg"
    save_image(panorama_path, result.panorama)
    times[5] = time.perf_counter() - start

    h, w = result.panorama.shape[:2]
    print_table([
        {"Item": "imagens na entrada", "valor": len(image_set)},
        {"Item": "sequência inferida", "valor": " -> ".join(image_set.names[k] for k in sorting.order)},
        {"Item": "intrusas rejeitadas", "valor": ", ".join(image_set.names[k] for k in sorting.rejected) or "nenhuma"},
        {"Item": "imagem de referência", "valor": image_set.names[sorting.reference]},
        {"Item": "projeção", "valor": config.projection + (f" (focal {alignment.focal:.0f} px)" if alignment.focal else "")},
        {"Item": "panorama", "valor": f"{w}x{h} px, {compositor.deghost} + {compositor.blend}"},
    ], f"Resumo do panorama ({group})")
    return finish(group, times, panorama_path)


def print_capture_metadata(folder: Path, group: str) -> None:
    """Shows input/<grupo>/metadados.json, when present. It is only displayed, never used."""
    path = folder / "metadados.json"
    if not path.exists():
        return
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:  # only displayed: an invalid file must not stop the pipeline
        log("Entrada", f"aviso: {path} não é um JSON válido ({error}); metadados não exibidos")
        return
    rows = []
    for key, value in metadata.items():
        if isinstance(value, dict):
            rows += [{"campo": f"{key}.{k}", "valor": v} for k, v in value.items()]
        elif isinstance(value, list):
            rows += [{"campo": key if n == 0 else "", "valor": v} for n, v in enumerate(value)]
        else:
            rows.append({"campo": key, "valor": value})
    print_table(rows, f"Metadados da coleta ({group})")


def finish(group: str, times: dict[int, float], panorama_path: Path | None = None) -> Path | None:
    print_table([{"passo": f"{n}: {STEPS[n]}", "tempo_s": t} for n, t in sorted(times.items())],
                f"Tempo por passo ({group})")
    log("Pipeline", f"panorama salvo em {panorama_path}" if panorama_path else
        f"execução interrompida após o passo {max(times)}; saídas em output/<passo>/{group}/")
    return panorama_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera o panorama de uma pasta de imagens fora de ordem.")
    parser.add_argument("pasta", nargs="?", help="pasta do grupo de imagens, ex.: input/paisagem")
    parser.add_argument("--ate-passo", type=int, default=5, choices=sorted(STEPS),
                        help="executa os passos 1 até N (padrão: todos)")
    parser.add_argument("--todos", action="store_true", help="processa todas as pastas de input/")
    args = parser.parse_args()
    if args.todos:
        folders = [d for d in sorted(INPUT_DIR.iterdir()) if d.is_dir() and list_image_files(d)]
    elif args.pasta:
        folders = [Path(args.pasta)]
    else:
        parser.error("informe a pasta de um grupo ou use --todos")
    for folder in folders:
        run(folder, args.ate_passo)


if __name__ == "__main__":
    main()
