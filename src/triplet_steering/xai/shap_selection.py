from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from triplet_steering.config import ExperimentConfig
from triplet_steering.models.metadata import MetadataOnlyRegressor


def train_metadata_only_model(
    cfg: ExperimentConfig,
    train_meta: np.ndarray,
    train_targets: np.ndarray,
    val_meta: np.ndarray,
    val_targets: np.ndarray,
    device: str,
    epochs: int | None = None,
    lr: float | None = None,
) -> MetadataOnlyRegressor:
    epochs = cfg.metadata_only_epochs if epochs is None else epochs
    lr = cfg.metadata_only_lr if lr is None else lr

    model = MetadataOnlyRegressor(
        input_dim=train_meta.shape[1],
        hidden_dims=cfg.metadata_only_hidden_dims,
        dropout=0.1,
    ).to(device)

    train_x = torch.tensor(train_meta, dtype=torch.float32)
    train_y = torch.tensor(train_targets, dtype=torch.float32)
    val_x = torch.tensor(val_meta, dtype=torch.float32)
    val_y = torch.tensor(val_targets, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(256, cfg.batch_size * 8),
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.05)

    best_state = None
    best_val_mae = float("inf")
    patience_left = 40

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for inputs, target in train_loader:
            inputs = inputs.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs).squeeze(1)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * inputs.size(0)
            n_seen += inputs.size(0)

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x.to(device)).squeeze(1)
            val_mae = F.l1_loss(val_prediction, val_y.to(device)).item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = 40
        else:
            patience_left -= 1

        print(f"[metadata-only] epoch {epoch:03d} | train_loss={running_loss / max(n_seen, 1):.5f} | val_mae={val_mae:.5f}")

        if patience_left <= 0:
            print("Metadata-only early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def rank_metadata_features_with_shap(
    cfg: ExperimentConfig,
    model: MetadataOnlyRegressor,
    train_meta: np.ndarray,
    val_meta: np.ndarray,
    feature_names: list[str],
    device: str,
    top_k: int | None = None,
):
    top_k = cfg.shap_top_k if top_k is None else top_k

    background_size = min(cfg.shap_background_size, len(train_meta))
    eval_size = min(cfg.shap_eval_size, len(val_meta))

    background = torch.tensor(train_meta[:background_size], dtype=torch.float32, device=device)
    eval_x = torch.tensor(val_meta[:eval_size], dtype=torch.float32, device=device)

    model.eval()
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(eval_x, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]
        elif shap_values.shape[1] == 1:
            shap_values = shap_values[:, 0, :]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    ranking_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top_k = min(top_k, len(ranking_df))
    selected_features = ranking_df.head(top_k)["feature"].tolist()
    selected_indices = [feature_names.index(feature) for feature in selected_features]
    selected_weights_raw = ranking_df.head(top_k)["mean_abs_shap"].to_numpy(dtype=np.float32)

    if np.all(selected_weights_raw <= 0):
        selected_weights = np.ones_like(selected_weights_raw, dtype=np.float32)
    else:
        selected_weights = selected_weights_raw / (selected_weights_raw.max() + 1e-8)

    return {
        "ranking_df": ranking_df,
        "selected_features": selected_features,
        "selected_indices": selected_indices,
        "selected_weights": selected_weights.astype(np.float32),
        "shap_values": shap_values,
        "eval_x": eval_x.detach().cpu().numpy(),
    }
