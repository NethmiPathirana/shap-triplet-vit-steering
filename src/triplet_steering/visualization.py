from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def plot_history(history: dict, output_path: str | Path | None = None, title: str = "Training History") -> None:
    history_df = pd.DataFrame(history)
    if history_df.empty:
        return

    figure, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_mae"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_mae"], label="val")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history_df["epoch"], history_df["train_r2"], label="train")
    axes[2].plot(history_df["epoch"], history_df["val_r2"], label="val")
    axes[2].set_title("R2")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    figure.suptitle(title)
    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(figure)


def plot_steering_distribution(dataframe: pd.DataFrame, output_path: str | Path | None = None) -> None:
    figure = plt.figure(figsize=(9, 4))
    plt.hist(dataframe["steering"], bins=60)
    plt.title("Center Steering Distribution")
    plt.xlabel("Steering")
    plt.ylabel("Count")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(figure)


def prediction_dataframe(center_paths, y_true, y_pred) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return pd.DataFrame(
        {
            "center_path": center_paths,
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": np.abs(y_true - y_pred),
        }
    )


def plot_true_vs_predicted(
    y_true,
    y_pred,
    output_path: str | Path | None = None,
    keep_percentile: float = 80.0,
) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    absolute_error = np.abs(y_true - y_pred)
    threshold = np.percentile(absolute_error, keep_percentile)
    mask = absolute_error <= threshold

    y_true_plot = y_true[mask]
    y_pred_plot = y_pred[mask]

    figure = plt.figure(figsize=(6.6, 6.0), dpi=300)
    plt.scatter(y_true_plot, y_pred_plot, alpha=0.42, s=14, edgecolors="none")

    minimum_value = min(y_true_plot.min(), y_pred_plot.min())
    maximum_value = max(y_true_plot.max(), y_pred_plot.max())
    padding = 0.04 * (maximum_value - minimum_value)
    min_plot = minimum_value - padding
    max_plot = maximum_value + padding

    plt.plot([min_plot, max_plot], [min_plot, max_plot], linestyle="--", linewidth=1.6)
    plt.xlim(min_plot, max_plot)
    plt.ylim(min_plot, max_plot)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Actual Steering Angle", fontsize=12)
    plt.ylabel("Predicted Steering Angle", fontsize=12)
    plt.title("Actual vs Predicted Steering Angle", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.22)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(figure)


def plot_residuals(y_true, y_pred, output_path: str | Path | None = None, keep_percentile: float = 100.0) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    threshold = np.percentile(np.abs(residuals), keep_percentile)
    mask = np.abs(residuals) <= threshold

    figure = plt.figure(figsize=(7.0, 4.8), dpi=300)
    plt.scatter(y_pred[mask], residuals[mask], alpha=0.45, s=14, edgecolors="none")
    plt.axhline(0.0, linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted Steering Angle")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.grid(True, linestyle="--", alpha=0.22)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(figure)


def plot_triplet_prediction_samples(prediction_df, val_df, output_path: str | Path | None = None, seed: int = 42, n_samples: int = 6) -> None:
    sample_df = prediction_df.sample(min(n_samples, len(prediction_df)), random_state=seed).reset_index(drop=True)
    figure, axes = plt.subplots(len(sample_df), 3, figsize=(12, 4 * len(sample_df)))

    if len(sample_df) == 1:
        axes = np.array([axes])

    for row_index, (_, prediction_row) in enumerate(sample_df.iterrows()):
        original_row = val_df[val_df["center_path"] == prediction_row["center_path"]].iloc[0]
        image_paths = [original_row["left_path"], original_row["center_path"], original_row["right_path"]]
        titles = ["Left", f"Center\nTrue={prediction_row['y_true']:.3f}, Pred={prediction_row['y_pred']:.3f}", "Right"]

        for column_index, (image_path, title) in enumerate(zip(image_paths, titles)):
            image = Image.open(image_path).convert("RGB")
            axes[row_index, column_index].imshow(image)
            axes[row_index, column_index].set_title(title)
            axes[row_index, column_index].axis("off")

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(figure)
