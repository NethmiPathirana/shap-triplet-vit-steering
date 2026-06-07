from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    data_dir: str = "data"
    cleaned_csv: str = "data/cleaned_driving_log.csv"
    checkpoints_dir: str = "outputs/checkpoints"

    seed: int = 220
    epochs: int = 150
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 15
    grad_clip: float = 1.0
    num_workers: int = 2
    use_amp: bool = True
    val_size: float = 0.20

    backbone_name: str = "vit_small_patch16_224"
    img_size: int = 224
    fusion_dim: int = 384
    fusion_dropout: float = 0.1

    sample_limit: int | None = None
    verify_images_again: bool = False

    numeric_meta_cols: tuple[str, ...] = ("throttle", "reverse", "speed")
    categorical_meta_cols: tuple[str, ...] = ()

    use_weighted_sampler: bool = True
    steering_bin_edges: tuple[float, ...] = (-1.0, -0.35, -0.15, -0.05, 0.05, 0.15, 0.35, 1.0)

    model_name: str = "triplet_vit_selected_meta_concat_center_steering"
    shap_background_size: int = 256
    shap_eval_size: int = 256
    shap_top_k: int = 2
    metadata_only_hidden_dims: tuple[int, ...] = (64, 32)
    metadata_only_epochs: int = 100
    metadata_only_lr: float = 1e-3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def ensure_directories(self) -> None:
        Path(self.checkpoints_dir).mkdir(parents=True, exist_ok=True)


def _coerce_tuple(value: Any, item_type: type) -> tuple:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(item_type(x) for x in value)
    return tuple(item_type(x.strip()) for x in str(value).split(",") if x.strip())


def load_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    data: dict[str, Any] = {}

    if path:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Configuration file must contain a YAML mapping.")
        data.update(loaded)

    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    valid_names = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(data) - valid_names)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")

    tuple_fields = {
        "numeric_meta_cols": str,
        "categorical_meta_cols": str,
        "steering_bin_edges": float,
        "metadata_only_hidden_dims": int,
    }

    for name, item_type in tuple_fields.items():
        if name in data:
            data[name] = _coerce_tuple(data[name], item_type)

    cfg = ExperimentConfig(**data)
    cfg.ensure_directories()
    return cfg
