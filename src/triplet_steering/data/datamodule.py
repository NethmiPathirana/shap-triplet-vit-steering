from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from triplet_steering.config import ExperimentConfig
from triplet_steering.data.dataset import TripletDrivingDataset
from triplet_steering.data.transforms import TripletTransform


def build_weighted_sampler(train_df: pd.DataFrame, bin_edges: tuple[float, ...]) -> WeightedRandomSampler:
    target = train_df["steering"].values.astype(np.float32)
    bins = np.digitize(target, bin_edges[1:-1], right=False)
    counts = np.bincount(bins, minlength=len(bin_edges) - 1).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[bins]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights),
        replacement=True,
    )


def build_datasets(train_df, val_df, train_meta, val_meta, cfg: ExperimentConfig):
    train_transform = TripletTransform(img_size=cfg.img_size, train=True)
    val_transform = TripletTransform(img_size=cfg.img_size, train=False)
    train_dataset = TripletDrivingDataset(train_df, train_meta, train_transform)
    val_dataset = TripletDrivingDataset(val_df, val_meta, val_transform)
    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, train_df, cfg: ExperimentConfig):
    sampler = build_weighted_sampler(train_df, cfg.steering_bin_edges) if cfg.use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
