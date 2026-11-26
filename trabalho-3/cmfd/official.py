"""Adaptador para a métrica oficial (metric/recodai_f1.py, sem modificações)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "metric"))
import recodai_f1 as _m  # noqa: E402


def encode_prediction(masks: list) -> str:
    """Lista de máscaras binárias -> string de submissão ('authentic' se vazia)."""
    masks = [np.asarray(m).astype(np.uint8) for m in masks if np.asarray(m).any()]
    return _m.rle_encode(masks) if masks else "authentic"


def score_one(gt_annotation: str, pred_annotation: str, shape) -> float:
    """S de uma única imagem, pela função oficial."""
    shp = json.dumps([int(shape[0]), int(shape[1])]) if gt_annotation != "authentic" else "authentic"
    sol = pd.DataFrame({"row_id": ["x"], "annotation": [gt_annotation], "shape": [shp]})
    sub = pd.DataFrame({"row_id": ["x"], "annotation": [pred_annotation]})
    return _m.score(sol, sub, row_id_column_name="row_id")
