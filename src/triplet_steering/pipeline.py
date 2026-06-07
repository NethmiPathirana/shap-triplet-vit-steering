from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triplet_steering.config import ExperimentConfig
from triplet_steering.data.datamodule import build_dataloaders, build_datasets
from triplet_steering.data.preprocessing import MetaPreprocessor, load_cleaned_dataframe, split_dataframe


@dataclass
class PreparedData:
    dataframe: object
    train_df: object
    val_df: object
    processor: MetaPreprocessor
    train_meta_full: np.ndarray
    val_meta_full: np.ndarray
    all_meta_feature_names: list[str]


@dataclass
class PreparedSelectedData:
    prepared: PreparedData
    train_meta: np.ndarray
    val_meta: np.ndarray
    selected_indices: list[int]
    selected_features: list[str]
    selected_weights: np.ndarray


def prepare_data(cfg: ExperimentConfig) -> PreparedData:
    dataframe = load_cleaned_dataframe(cfg)
    train_df, val_df = split_dataframe(dataframe, cfg)

    processor = MetaPreprocessor(cfg.numeric_meta_cols, cfg.categorical_meta_cols)
    processor.fit(train_df)

    train_meta_full = processor.transform(train_df)
    val_meta_full = processor.transform(val_df)
    all_meta_feature_names = list(processor.feature_names_)

    return PreparedData(
        dataframe=dataframe,
        train_df=train_df,
        val_df=val_df,
        processor=processor,
        train_meta_full=train_meta_full,
        val_meta_full=val_meta_full,
        all_meta_feature_names=all_meta_feature_names,
    )


def prepare_selected_data_from_checkpoint(cfg: ExperimentConfig, checkpoint: dict) -> PreparedSelectedData:
    dataframe = load_cleaned_dataframe(cfg)
    train_df, val_df = split_dataframe(dataframe, cfg)

    metadata_context = checkpoint.get("metadata_context", {})
    processor_state = metadata_context.get("meta_processor")

    if processor_state is None:
        numeric_mean = metadata_context.get("meta_numeric_mean", {})
        cat_levels = metadata_context.get("meta_cat_levels", {})
        processor_state = {
            "numeric_cols": metadata_context.get("numeric_cols", list(numeric_mean.keys()) or cfg.numeric_meta_cols),
            "categorical_cols": metadata_context.get("categorical_cols", list(cat_levels.keys()) or cfg.categorical_meta_cols),
            "numeric_mean": numeric_mean,
            "numeric_std": metadata_context.get("meta_numeric_std", {}),
            "cat_levels": cat_levels,
            "feature_names": metadata_context.get("all_meta_feature_names", []),
        }

    processor = MetaPreprocessor.from_state_dict(processor_state)
    train_meta_full = processor.transform(train_df)
    val_meta_full = processor.transform(val_df)

    selected_indices = metadata_context.get("selected_meta_indices")
    if selected_indices is None:
        selected_indices = metadata_context.get("selected_indices")

    selected_features = metadata_context.get("selected_meta_feature_names")
    if selected_features is None:
        selected_features = metadata_context.get("selected_features")

    selected_weights = metadata_context.get("selected_meta_shap_weights")
    if selected_weights is None:
        selected_weights = metadata_context.get("selected_weights")

    if selected_indices is None:
        if selected_features is None:
            raise KeyError("Checkpoint does not contain selected metadata indices or selected metadata features.")
        selected_indices = [processor.feature_names_.index(feature) for feature in selected_features]

    if selected_features is None:
        selected_features = [processor.feature_names_[index] for index in selected_indices]

    if selected_weights is None:
        selected_weights = np.ones((len(selected_indices),), dtype=np.float32)

    prepared = PreparedData(
        dataframe=dataframe,
        train_df=train_df,
        val_df=val_df,
        processor=processor,
        train_meta_full=train_meta_full,
        val_meta_full=val_meta_full,
        all_meta_feature_names=list(processor.feature_names_),
    )

    return PreparedSelectedData(
        prepared=prepared,
        train_meta=train_meta_full[:, selected_indices].astype(np.float32),
        val_meta=val_meta_full[:, selected_indices].astype(np.float32),
        selected_indices=list(selected_indices),
        selected_features=list(selected_features),
        selected_weights=np.asarray(selected_weights, dtype=np.float32),
    )


def build_loaders_for_selected_data(selected_data: PreparedSelectedData, cfg: ExperimentConfig):
    train_dataset, val_dataset = build_datasets(
        selected_data.prepared.train_df,
        selected_data.prepared.val_df,
        selected_data.train_meta,
        selected_data.val_meta,
        cfg,
    )
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, selected_data.prepared.train_df, cfg)
    return train_dataset, val_dataset, train_loader, val_loader


def metadata_context_from_selection(prepared: PreparedData, selected_indices: list[int], selected_features: list[str], selected_weights) -> dict:
    return {
        "selected_meta_feature_names": list(selected_features),
        "selected_meta_indices": [int(index) for index in selected_indices],
        "selected_meta_shap_weights": np.asarray(selected_weights, dtype=np.float32).tolist(),
        "all_meta_feature_names": list(prepared.all_meta_feature_names),
        "meta_processor": prepared.processor.state_dict(),
    }
