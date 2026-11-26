"""Adaptador para a métrica OFICIAL (metric/recodai_f1.py, sem modificações)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "metric"))
import recodai_f1 as _m  # noqa: E402

ROW_ID = "row_id"


def encode_prediction(masks: list) -> str:
    """Lista de máscaras binárias -> string de submissão ('authentic' se vazia)."""
    masks = [np.asarray(m).astype(np.uint8) for m in masks if np.asarray(m).any()]
    if not masks:
        return "authentic"
    return _m.rle_encode(masks)


def solution_frame(keys, annotations, shapes) -> pd.DataFrame:
    shp = [json.dumps([int(h), int(w)]) if a != "authentic" else "authentic"
           for a, (h, w) in zip(annotations, shapes)]
    return pd.DataFrame({ROW_ID: list(keys), "annotation": list(annotations), "shape": shp})


def submission_frame(keys, annotations) -> pd.DataFrame:
    return pd.DataFrame({ROW_ID: list(keys), "annotation": list(annotations)})


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    sol = solution.reset_index(drop=True).copy()
    sub = submission.reset_index(drop=True).copy()
    assert (sol[ROW_ID].to_numpy() == sub[ROW_ID].to_numpy()).all(), "gabarito e previsão fora de ordem"
    return _m.score(sol, sub, row_id_column_name=ROW_ID)


def score_one(gt_annotation: str, pred_annotation: str, shape) -> float:
    """S de uma única imagem, pela mesma função oficial."""
    sol = solution_frame(["x"], [gt_annotation], [shape])
    sub = submission_frame(["x"], [pred_annotation])
    return _m.score(sol, sub, row_id_column_name=ROW_ID)
