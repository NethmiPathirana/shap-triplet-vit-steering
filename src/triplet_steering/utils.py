from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_rgb_image(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return False
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB")
        return True
    except (UnidentifiedImageError, OSError, IOError, SyntaxError, ValueError):
        return False


def clamp_steering(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def direction_from_steering(value: float, eps: float = 0.05) -> str:
    if value < -eps:
        return "left"
    if value > eps:
        return "right"
    return "straight"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(np.float32)
    y_pred = np.asarray(y_pred).astype(np.float32)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    direction_accuracy = np.mean(
        [
            direction_from_steering(true_value) == direction_from_steering(pred_value)
            for true_value, pred_value in zip(y_true, y_pred)
        ]
    )
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "dir_acc": float(direction_accuracy),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_dataframe_csv(dataframe, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
