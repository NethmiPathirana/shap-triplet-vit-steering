from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from triplet_steering.config import ExperimentConfig
from triplet_steering.data.preprocessing import MetaPreprocessor
from triplet_steering.models.triplet_vit import TripletViTMetaRegressor


def clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "", 1)
        if new_key.startswith("_orig_mod."):
            new_key = new_key.replace("_orig_mod.", "", 1)
        cleaned[new_key] = value
    return cleaned


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in checkpoint:
                return clean_state_dict(checkpoint[key])
    return clean_state_dict(checkpoint)


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_mae: float,
    history: dict,
    cfg: ExperimentConfig,
    metadata_context: dict,
) -> None:
    payload = {
        "epoch": epoch,
        "best_val_mae": best_val_mae,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "cfg": cfg.to_dict(),
        "metadata_context": metadata_context,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _first_present(mapping: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def build_model_from_checkpoint(cfg: ExperimentConfig, checkpoint: dict, device: str, pretrained: bool = False):
    metadata_context = checkpoint.get("metadata_context", {})
    selected_features = _first_present(metadata_context, ("selected_meta_feature_names", "selected_features"))
    selected_weights = _first_present(metadata_context, ("selected_meta_shap_weights", "selected_weights"))

    if selected_features is None:
        selected_indices = _first_present(metadata_context, ("selected_meta_indices", "selected_indices"))
        if selected_indices is None:
            raise KeyError("Checkpoint does not contain selected metadata features or indices.")
        meta_dim = len(selected_indices)
    else:
        meta_dim = len(selected_features)

    if selected_weights is None:
        selected_weights = np.ones((meta_dim,), dtype=np.float32)

    model = TripletViTMetaRegressor(
        backbone_name=cfg.backbone_name,
        meta_dim=meta_dim,
        fusion_dim=cfg.fusion_dim,
        dropout=cfg.fusion_dropout,
        meta_shap_weights=np.asarray(selected_weights, dtype=np.float32),
        pretrained=pretrained,
    ).to(device)

    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()
    return model


def metadata_processor_from_checkpoint(checkpoint: dict) -> MetaPreprocessor:
    metadata_context = checkpoint.get("metadata_context", {})
    processor_state = metadata_context.get("meta_processor")
    if processor_state is None:
        numeric_mean = metadata_context.get("meta_numeric_mean", {})
        cat_levels = metadata_context.get("meta_cat_levels", {})
        processor_state = {
            "numeric_cols": metadata_context.get("numeric_cols", list(numeric_mean.keys())),
            "categorical_cols": metadata_context.get("categorical_cols", list(cat_levels.keys())),
            "numeric_mean": numeric_mean,
            "numeric_std": metadata_context.get("meta_numeric_std", {}),
            "cat_levels": cat_levels,
            "feature_names": metadata_context.get("all_meta_feature_names", []),
        }
    return MetaPreprocessor.from_state_dict(processor_state)
