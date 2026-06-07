from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from triplet_steering.config import ExperimentConfig
from triplet_steering.utils import clamp_steering, verify_rgb_image


REQUIRED_COLUMNS = ["left_path", "center_path", "right_path", "steering", "throttle", "reverse", "speed"]
PATH_COLUMNS = ["left_path", "center_path", "right_path"]


class MetaPreprocessor:
    def __init__(self, numeric_cols: tuple[str, ...], categorical_cols: tuple[str, ...]):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.numeric_mean_: dict[str, float] = {}
        self.numeric_std_: dict[str, float] = {}
        self.cat_levels_: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []

    def fit(self, dataframe: pd.DataFrame) -> None:
        self.feature_names_ = []

        for column in self.numeric_cols:
            values = pd.to_numeric(dataframe[column], errors="coerce").fillna(0.0).astype(np.float32)
            mean = float(values.mean())
            std = float(values.std())
            if std < 1e-8:
                std = 1.0
            self.numeric_mean_[column] = mean
            self.numeric_std_[column] = std
            self.feature_names_.append(f"{column}_z")

        for column in self.categorical_cols:
            values = dataframe[column].fillna("unknown").astype(str)
            categories = sorted(values.unique().tolist())
            if "unknown" not in categories:
                categories.append("unknown")
            self.cat_levels_[column] = categories
            self.feature_names_.extend([f"{column}={category}" for category in categories])

    def transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        arrays = []

        for column in self.numeric_cols:
            values = pd.to_numeric(dataframe[column], errors="coerce").fillna(self.numeric_mean_[column]).astype(np.float32)
            values = ((values - self.numeric_mean_[column]) / self.numeric_std_[column]).values.reshape(-1, 1)
            arrays.append(values.astype(np.float32))

        for column in self.categorical_cols:
            values = dataframe[column].fillna("unknown").astype(str).values
            categories = self.cat_levels_[column]
            category_to_index = {category: index for index, category in enumerate(categories)}
            array = np.zeros((len(values), len(categories)), dtype=np.float32)
            unknown_index = category_to_index.get("unknown", 0)
            for row_index, value in enumerate(values):
                array[row_index, category_to_index.get(value, unknown_index)] = 1.0
            arrays.append(array)

        if arrays:
            return np.concatenate(arrays, axis=1).astype(np.float32)
        return np.zeros((len(dataframe), 0), dtype=np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "numeric_mean": self.numeric_mean_,
            "numeric_std": self.numeric_std_,
            "cat_levels": self.cat_levels_,
            "feature_names": self.feature_names_,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "MetaPreprocessor":
        processor = cls(tuple(state["numeric_cols"]), tuple(state["categorical_cols"]))
        processor.numeric_mean_ = dict(state["numeric_mean"])
        processor.numeric_std_ = dict(state["numeric_std"])
        processor.cat_levels_ = {key: list(value) for key, value in state["cat_levels"].items()}
        processor.feature_names_ = list(state["feature_names"])
        return processor


def _resolve_path(value: str, data_dir: str) -> str:
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str(Path(data_dir) / path)


def resolve_image_paths(dataframe: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for column in PATH_COLUMNS:
        dataframe[column] = dataframe[column].astype(str).apply(lambda value: _resolve_path(value, data_dir))
    return dataframe


def load_cleaned_dataframe(cfg: ExperimentConfig) -> pd.DataFrame:
    csv_path = Path(cfg.cleaned_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "track" not in dataframe.columns:
        dataframe["track"] = "unknown_track"

    if "row_id" not in dataframe.columns:
        dataframe["row_id"] = np.arange(len(dataframe))

    dataframe = resolve_image_paths(dataframe, cfg.data_dir)
    dataframe["steering"] = dataframe["steering"].apply(clamp_steering).astype(np.float32)

    for column in ["throttle", "reverse", "speed"]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0.0).astype(np.float32)

    dataframe = dataframe.dropna(subset=PATH_COLUMNS + ["steering"]).reset_index(drop=True)

    if cfg.verify_images_again:
        valid_mask = []
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Verifying images"):
            valid_mask.append(all(verify_rgb_image(row[column]) for column in PATH_COLUMNS))
        dataframe = dataframe[np.asarray(valid_mask, dtype=bool)].reset_index(drop=True)

    if cfg.sample_limit is not None:
        sample_size = min(int(cfg.sample_limit), len(dataframe))
        dataframe = dataframe.sample(sample_size, random_state=cfg.seed).reset_index(drop=True)

    return dataframe


def split_dataframe(dataframe: pd.DataFrame, cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=cfg.seed)
    train_index, val_index = next(splitter.split(dataframe, groups=dataframe["row_id"]))
    train_df = dataframe.iloc[train_index].reset_index(drop=True)
    val_df = dataframe.iloc[val_index].reset_index(drop=True)
    return train_df, val_df


def fit_metadata_processor(train_df: pd.DataFrame, cfg: ExperimentConfig) -> tuple[MetaPreprocessor, np.ndarray, list[str]]:
    processor = MetaPreprocessor(cfg.numeric_meta_cols, cfg.categorical_meta_cols)
    processor.fit(train_df)
    train_meta_full = processor.transform(train_df)
    return processor, train_meta_full, list(processor.feature_names_)
