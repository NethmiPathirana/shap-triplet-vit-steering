from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from triplet_steering.config import load_config
from triplet_steering.data.datamodule import build_dataloaders, build_datasets
from triplet_steering.models.triplet_vit import TripletViTMetaRegressor
from triplet_steering.pipeline import metadata_context_from_selection, prepare_data
from triplet_steering.training.checkpoints import build_model_from_checkpoint, load_checkpoint
from triplet_steering.training.engine import evaluate, train_model
from triplet_steering.utils import get_device, save_dataframe_csv, save_json, set_seed
from triplet_steering.visualization import (
    plot_history,
    plot_residuals,
    plot_steering_distribution,
    plot_triplet_prediction_samples,
    plot_true_vs_predicted,
    prediction_dataframe,
)
from triplet_steering.xai.shap_selection import rank_metadata_features_with_shap, train_metadata_only_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the triplet multimodal ViT steering regression model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device()

    print(f"Device: {device}")
    print(f"Checkpoint directory: {cfg.checkpoints_dir}")

    prepared = prepare_data(cfg)

    print(f"Loaded rows: {len(prepared.dataframe)}")
    print(f"Train rows: {len(prepared.train_df)}")
    print(f"Validation rows: {len(prepared.val_df)}")
    print(f"Full metadata dimension: {prepared.train_meta_full.shape[1]}")
    print(f"Metadata features: {prepared.all_meta_feature_names}")

    figures_dir = Path(cfg.checkpoints_dir) / "figures"
    artifacts_dir = Path(cfg.checkpoints_dir) / "artifacts"
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_plots:
        plot_steering_distribution(prepared.dataframe, figures_dir / "steering_distribution.png")

    train_targets = prepared.train_df["steering"].astype(np.float32).values
    val_targets = prepared.val_df["steering"].astype(np.float32).values

    metadata_only_model = train_metadata_only_model(
        cfg=cfg,
        train_meta=prepared.train_meta_full,
        train_targets=train_targets,
        val_meta=prepared.val_meta_full,
        val_targets=val_targets,
        device=device,
    )

    shap_result = rank_metadata_features_with_shap(
        cfg=cfg,
        model=metadata_only_model,
        train_meta=prepared.train_meta_full,
        val_meta=prepared.val_meta_full,
        feature_names=prepared.all_meta_feature_names,
        device=device,
        top_k=cfg.shap_top_k,
    )

    ranking_df = shap_result["ranking_df"]
    selected_features = shap_result["selected_features"]
    selected_indices = shap_result["selected_indices"]
    selected_weights = shap_result["selected_weights"]

    save_dataframe_csv(ranking_df, artifacts_dir / "metadata_shap_ranking.csv")
    save_json(
        artifacts_dir / "selected_metadata.json",
        {
            "selected_features": selected_features,
            "selected_indices": selected_indices,
            "selected_weights": selected_weights.tolist(),
        },
    )

    train_meta = prepared.train_meta_full[:, selected_indices].astype(np.float32)
    val_meta = prepared.val_meta_full[:, selected_indices].astype(np.float32)

    train_dataset, val_dataset = build_datasets(prepared.train_df, prepared.val_df, train_meta, val_meta, cfg)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, prepared.train_df, cfg)

    metadata_context = metadata_context_from_selection(prepared, selected_indices, selected_features, selected_weights)

    model = TripletViTMetaRegressor(
        backbone_name=cfg.backbone_name,
        meta_dim=train_meta.shape[1],
        fusion_dim=cfg.fusion_dim,
        dropout=cfg.fusion_dropout,
        meta_shap_weights=selected_weights,
        pretrained=True,
    )

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        metadata_context=metadata_context,
    )

    save_dataframe_csv(pd.DataFrame(result["history"]), artifacts_dir / "training_history.csv")

    if not args.no_plots:
        plot_history(result["history"], figures_dir / "training_history.png", title="Triplet ViT + Metadata Model")

    best_checkpoint = load_checkpoint(result["best_path"], device=device)
    best_model = build_model_from_checkpoint(cfg, best_checkpoint, device=device, pretrained=False)
    val_loss, val_metrics, y_true, y_pred, center_paths = evaluate(best_model, val_loader, device=device)

    predictions = prediction_dataframe(center_paths, y_true, y_pred)
    save_dataframe_csv(predictions, artifacts_dir / "validation_predictions.csv")
    save_json(artifacts_dir / "validation_metrics.json", {"val_loss": val_loss, **val_metrics})

    if not args.no_plots:
        plot_true_vs_predicted(y_true, y_pred, figures_dir / "actual_vs_predicted.png")
        plot_residuals(y_true, y_pred, figures_dir / "residuals.png")
        plot_triplet_prediction_samples(predictions, prepared.val_df, figures_dir / "sample_predictions.png", seed=cfg.seed)

    print("Training completed.")
    print(f"Best checkpoint: {result['best_path']}")
    print(f"Best validation MAE: {result['best_val_mae']:.6f}")


if __name__ == "__main__":
    main()
