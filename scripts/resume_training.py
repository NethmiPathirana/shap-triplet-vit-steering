from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from triplet_steering.config import load_config
from triplet_steering.pipeline import build_loaders_for_selected_data, prepare_selected_data_from_checkpoint
from triplet_steering.training.checkpoints import build_model_from_checkpoint, load_checkpoint
from triplet_steering.training.engine import continue_training
from triplet_steering.utils import get_device, save_dataframe_csv, set_seed
from triplet_steering.visualization import plot_history


def parse_args():
    parser = argparse.ArgumentParser(description="Resume training from a saved checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--extra-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device()

    checkpoint = load_checkpoint(args.checkpoint, device=device)
    selected_data = prepare_selected_data_from_checkpoint(cfg, checkpoint)
    _, _, train_loader, val_loader = build_loaders_for_selected_data(selected_data, cfg)

    metadata_context = checkpoint["metadata_context"]
    model = build_model_from_checkpoint(cfg, checkpoint, device=device, pretrained=False)
    model, history = continue_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        metadata_context=metadata_context,
        checkpoint=checkpoint,
        extra_epochs=args.extra_epochs,
        patience=args.patience,
    )

    output_dir = Path(cfg.checkpoints_dir) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dataframe_csv(pd.DataFrame(history), output_dir / "resumed_training_history.csv")
    plot_history(history, Path(cfg.checkpoints_dir) / "figures" / "resumed_training_history.png", title="Resumed Training History")

    print("Resume training completed.")


if __name__ == "__main__":
    main()
