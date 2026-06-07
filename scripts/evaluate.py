from __future__ import annotations

import argparse
from pathlib import Path

from triplet_steering.config import load_config
from triplet_steering.pipeline import build_loaders_for_selected_data, prepare_selected_data_from_checkpoint
from triplet_steering.training.checkpoints import build_model_from_checkpoint, load_checkpoint
from triplet_steering.training.engine import evaluate
from triplet_steering.utils import get_device, save_dataframe_csv, save_json, set_seed
from triplet_steering.visualization import plot_residuals, plot_true_vs_predicted, prediction_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained triplet multimodal ViT steering model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device()

    checkpoint = load_checkpoint(args.checkpoint, device=device)
    selected_data = prepare_selected_data_from_checkpoint(cfg, checkpoint)
    _, _, _, val_loader = build_loaders_for_selected_data(selected_data, cfg)

    model = build_model_from_checkpoint(cfg, checkpoint, device=device, pretrained=False)
    val_loss, val_metrics, y_true, y_pred, center_paths = evaluate(model, val_loader, device=device)

    output_dir = Path(cfg.checkpoints_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = prediction_dataframe(center_paths, y_true, y_pred)
    save_dataframe_csv(predictions, output_dir / "validation_predictions.csv")
    save_json(output_dir / "validation_metrics.json", {"val_loss": val_loss, **val_metrics})

    plot_true_vs_predicted(y_true, y_pred, output_dir / "actual_vs_predicted.png")
    plot_residuals(y_true, y_pred, output_dir / "residuals.png")

    print("Evaluation completed.")
    print(f"Validation loss: {val_loss:.6f}")
    print(val_metrics)


if __name__ == "__main__":
    main()
