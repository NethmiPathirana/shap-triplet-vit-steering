from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from triplet_steering.config import ExperimentConfig
from triplet_steering.training.checkpoints import save_checkpoint
from triplet_steering.utils import regression_metrics


def batch_to_device(batch: dict, device: str) -> dict:
    return {
        "left": batch["left"].to(device, non_blocking=True),
        "center": batch["center"].to(device, non_blocking=True),
        "right": batch["right"].to(device, non_blocking=True),
        "meta": batch["meta"].to(device, non_blocking=True),
        "target": batch["target"].to(device, non_blocking=True),
        "left_path": batch["left_path"],
        "center_path": batch["center_path"],
        "right_path": batch["right_path"],
    }


def forward_model(model, batch: dict):
    return model(batch["left"], batch["center"], batch["right"], batch["meta"])


def train_one_epoch(model, loader, optimizer, scaler, cfg: ExperimentConfig, device: str):
    criterion = nn.SmoothL1Loss(beta=0.05)
    model.train()
    running_loss = 0.0
    all_true = []
    all_pred = []

    use_amp = bool(cfg.use_amp and device == "cuda")

    for batch in tqdm(loader, leave=False):
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            prediction = forward_model(model, batch)
            loss = criterion(prediction, batch["target"])

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        running_loss += loss.item() * batch["target"].size(0)
        all_true.extend(batch["target"].detach().cpu().numpy())
        all_pred.extend(prediction.detach().cpu().numpy())

    average_loss = running_loss / len(loader.dataset)
    metrics = regression_metrics(all_true, np.clip(all_pred, -1.0, 1.0))
    return average_loss, metrics


@torch.no_grad()
def evaluate(model, loader, device: str):
    criterion = nn.MSELoss()
    model.eval()
    running_loss = 0.0
    all_true = []
    all_pred = []
    center_paths = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        prediction = forward_model(model, batch)
        loss = criterion(prediction, batch["target"])

        running_loss += loss.item() * batch["target"].size(0)
        all_true.extend(batch["target"].detach().cpu().numpy())
        all_pred.extend(prediction.detach().cpu().numpy())
        center_paths.extend(batch["center_path"])

    clipped_pred = np.clip(np.array(all_pred), -1.0, 1.0)
    average_loss = running_loss / len(loader.dataset)
    metrics = regression_metrics(all_true, clipped_pred)
    return average_loss, metrics, np.array(all_true), clipped_pred, center_paths


def new_history() -> dict:
    return {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_rmse": [],
        "val_rmse": [],
        "train_r2": [],
        "val_r2": [],
        "train_dir_acc": [],
        "val_dir_acc": [],
    }


def append_history(history: dict, epoch: int, train_loss: float, train_metrics: dict, val_loss: float, val_metrics: dict) -> None:
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_mae"].append(train_metrics["mae"])
    history["val_mae"].append(val_metrics["mae"])
    history["train_rmse"].append(train_metrics["rmse"])
    history["val_rmse"].append(val_metrics["rmse"])
    history["train_r2"].append(train_metrics["r2"])
    history["val_r2"].append(val_metrics["r2"])
    history["train_dir_acc"].append(train_metrics["dir_acc"])
    history["val_dir_acc"].append(val_metrics["dir_acc"])


def train_model(model, train_loader, val_loader, cfg: ExperimentConfig, device: str, metadata_context: dict, epochs: int | None = None):
    model = model.to(device)
    epochs = cfg.epochs if epochs is None else epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device == "cuda"))

    history = new_history()
    best_val_mae = float("inf")
    bad_epochs = 0

    best_path = Path(cfg.checkpoints_dir) / f"{cfg.model_name}_best.pth"
    last_path = Path(cfg.checkpoints_dir) / f"{cfg.model_name}_last.pth"

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device)
        val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        append_history(history, epoch, train_loss, train_metrics, val_loss, val_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} mae {train_metrics['mae']:.4f} rmse {train_metrics['rmse']:.4f} r2 {train_metrics['r2']:.4f} dir {train_metrics['dir_acc']:.4f} | "
            f"val loss {val_loss:.4f} mae {val_metrics['mae']:.4f} rmse {val_metrics['rmse']:.4f} r2 {val_metrics['r2']:.4f} dir {val_metrics['dir_acc']:.4f}"
        )

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val_mae, history, cfg, metadata_context)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_mae, history, cfg, metadata_context)
            print("Saved best checkpoint.")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print("Early stopping.")
                break

    return {
        "model": model,
        "history": history,
        "best_path": str(best_path),
        "last_path": str(last_path),
        "best_val_mae": best_val_mae,
    }


def continue_training(
    model,
    train_loader,
    val_loader,
    cfg: ExperimentConfig,
    device: str,
    metadata_context: dict,
    checkpoint: dict,
    extra_epochs: int,
    patience: int,
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=extra_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device == "cuda"))

    if checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if checkpoint.get("scheduler_state") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        except Exception:
            print("Scheduler state was not restored; using a fresh scheduler.")

    history = checkpoint.get("history", new_history())
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    end_epoch = start_epoch + extra_epochs - 1
    best_val_mae = float(checkpoint.get("best_val_mae", np.inf))
    bad_epochs = 0

    best_path = Path(cfg.checkpoints_dir) / f"{cfg.model_name}_best.pth"
    last_path = Path(cfg.checkpoints_dir) / f"{cfg.model_name}_last.pth"

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device)
        val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        append_history(history, epoch, train_loss, train_metrics, val_loss, val_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"train mae {train_metrics['mae']:.4f} rmse {train_metrics['rmse']:.4f} r2 {train_metrics['r2']:.4f} dir {train_metrics['dir_acc']:.4f} | "
            f"val mae {val_metrics['mae']:.4f} rmse {val_metrics['rmse']:.4f} r2 {val_metrics['r2']:.4f} dir {val_metrics['dir_acc']:.4f}"
        )

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val_mae, history, cfg, metadata_context)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_mae, history, cfg, metadata_context)
            print("Saved new best checkpoint.")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping during resumed training.")
                break

    return model, history
