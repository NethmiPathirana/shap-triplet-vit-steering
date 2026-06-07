from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, NoiseTunnel
from tqdm.auto import tqdm

from triplet_steering.data.transforms import IMAGENET_MEAN, IMAGENET_STD


class IGTripletWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, left, center, right, metadata):
        output = self.base_model(left, center, right, metadata)
        if output.dim() == 1:
            output = output.unsqueeze(1)
        return output


def unnormalize_to_np(image_tensor):
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    image = image_tensor.detach().cpu().float().permute(1, 2, 0).contiguous().numpy()
    image = image * std + mean
    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


def normalize_map(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.abs(values)
    values = values - values.min()
    values = values / (values.max() + 1e-8)
    return values


def img_attr_to_heatmap(attr_tensor):
    heatmap = attr_tensor.detach().cpu().abs().sum(dim=0).numpy()
    return normalize_map(heatmap)


def topk_metadata_table(meta_vec, meta_attr, feature_names: list[str], k: int = 10):
    metadata_df = pd.DataFrame(
        {
            "feature": feature_names,
            "value": meta_vec,
            "attribution": meta_attr,
            "abs_attribution": np.abs(meta_attr),
        }
    ).sort_values("abs_attribution", ascending=False)
    return metadata_df.head(k), metadata_df


def modality_contributions(attr_left, attr_center, attr_right, attr_meta):
    scores = {
        "left_image": float(attr_left.detach().abs().sum().item()),
        "center_image": float(attr_center.detach().abs().sum().item()),
        "right_image": float(attr_right.detach().abs().sum().item()),
        "metadata": float(attr_meta.detach().abs().sum().item()),
    }
    total = sum(scores.values()) + 1e-12
    percentages = {key: 100.0 * value / total for key, value in scores.items()}
    return scores, percentages


def build_captum_objects(model):
    wrapper = IGTripletWrapper(model)
    integrated_gradients = IntegratedGradients(wrapper)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    return integrated_gradients, noise_tunnel


def explain_one_triplet_sample(
    model,
    dataset,
    sample_idx: int,
    feature_names: list[str],
    device: str,
    output_dir: str | Path | None = None,
    use_noise_tunnel: bool = True,
    nt_samples: int = 6,
    n_steps: int = 24,
    show_top_meta: int = 10,
):
    model.eval()
    integrated_gradients, noise_tunnel = build_captum_objects(model)

    sample = dataset[sample_idx]
    left = sample["left"].unsqueeze(0).to(device)
    center = sample["center"].unsqueeze(0).to(device)
    right = sample["right"].unsqueeze(0).to(device)
    metadata = sample["meta"].unsqueeze(0).to(device)
    target = float(sample["target"].item())

    with torch.no_grad():
        prediction = float(model(left, center, right, metadata).item())

    baselines = (
        torch.zeros_like(left),
        torch.zeros_like(center),
        torch.zeros_like(right),
        torch.zeros_like(metadata),
    )

    if use_noise_tunnel:
        attrs, delta = noise_tunnel.attribute(
            inputs=(left, center, right, metadata),
            baselines=baselines,
            target=0,
            nt_type="smoothgrad_sq",
            nt_samples=nt_samples,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
    else:
        attrs, delta = integrated_gradients.attribute(
            inputs=(left, center, right, metadata),
            baselines=baselines,
            target=0,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

    attr_left, attr_center, attr_right, attr_meta = attrs

    left_np = unnormalize_to_np(left[0])
    center_np = unnormalize_to_np(center[0])
    right_np = unnormalize_to_np(right[0])

    heat_left = img_attr_to_heatmap(attr_left[0])
    heat_center = img_attr_to_heatmap(attr_center[0])
    heat_right = img_attr_to_heatmap(attr_right[0])

    meta_vec = metadata[0].detach().cpu().numpy()
    meta_attr = attr_meta[0].detach().cpu().numpy()

    if len(feature_names) != len(meta_vec) or len(feature_names) != len(meta_attr):
        raise ValueError("selected metadata feature names must match the final model metadata dimension.")

    top_meta_df, full_meta_df = topk_metadata_table(
        meta_vec=meta_vec,
        meta_attr=meta_attr,
        feature_names=feature_names,
        k=min(show_top_meta, len(feature_names)),
    )

    _, modality_percent = modality_contributions(attr_left[0], attr_center[0], attr_right[0], attr_meta[0])

    figure, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].imshow(left_np)
    axes[0, 0].set_title("Left Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(center_np)
    axes[0, 1].set_title(f"Center Image\nTrue={target:.3f}, Pred={prediction:.3f}")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(right_np)
    axes[0, 2].set_title("Right Image")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(left_np)
    axes[1, 0].imshow(heat_left, cmap="jet", alpha=0.45)
    axes[1, 0].set_title("Left Attribution")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(center_np)
    axes[1, 1].imshow(heat_center, cmap="jet", alpha=0.45)
    axes[1, 1].set_title("Center Attribution")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(right_np)
    axes[1, 2].imshow(heat_right, cmap="jet", alpha=0.45)
    axes[1, 2].set_title("Right Attribution")
    axes[1, 2].axis("off")

    figure.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir / f"xai_sample_{sample_idx}.png", dpi=300, bbox_inches="tight")
        top_meta_df.to_csv(output_dir / f"xai_sample_{sample_idx}_metadata.csv", index=False)

    plt.close(figure)

    return {
        "sample_idx": int(sample_idx),
        "true": target,
        "pred": prediction,
        "absolute_error": abs(target - prediction),
        "delta": float(torch.mean(torch.abs(delta)).item()),
        "top_metadata_df": top_meta_df,
        "full_metadata_df": full_meta_df,
        "modality_percent_df": pd.DataFrame(
            {
                "modality": list(modality_percent.keys()),
                "percent": list(modality_percent.values()),
            }
        ).sort_values("percent", ascending=False),
    }


def explain_multiple_samples(
    model,
    dataset,
    sample_indices,
    feature_names: list[str],
    device: str,
    output_dir: str | Path | None = None,
    use_noise_tunnel: bool = True,
    nt_samples: int = 6,
    n_steps: int = 24,
    show_top_meta: int = 10,
):
    results = []

    for sample_idx in sample_indices:
        result = explain_one_triplet_sample(
            model=model,
            dataset=dataset,
            sample_idx=int(sample_idx),
            feature_names=feature_names,
            device=device,
            output_dir=output_dir,
            use_noise_tunnel=use_noise_tunnel,
            nt_samples=nt_samples,
            n_steps=n_steps,
            show_top_meta=show_top_meta,
        )
        results.append(
            {
                "sample_idx": result["sample_idx"],
                "true": result["true"],
                "pred": result["pred"],
                "abs_error": result["absolute_error"],
                "delta": result["delta"],
            }
        )

    summary_df = pd.DataFrame(results).sort_values("abs_error", ascending=False).reset_index(drop=True)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "xai_summary.csv", index=False)

    return summary_df


def global_metadata_xai(model, dataset, feature_names: list[str], device: str, n_samples: int = 100, n_steps: int = 16, seed: int = 42):
    model.eval()
    integrated_gradients, _ = build_captum_objects(model)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    abs_meta_attrs = []
    modality_rows = []

    for sample_idx in tqdm(indices):
        sample = dataset[int(sample_idx)]
        left = sample["left"].unsqueeze(0).to(device)
        center = sample["center"].unsqueeze(0).to(device)
        right = sample["right"].unsqueeze(0).to(device)
        metadata = sample["meta"].unsqueeze(0).to(device)

        baselines = (
            torch.zeros_like(left),
            torch.zeros_like(center),
            torch.zeros_like(right),
            torch.zeros_like(metadata),
        )

        attrs = integrated_gradients.attribute(
            inputs=(left, center, right, metadata),
            baselines=baselines,
            target=0,
            n_steps=n_steps,
            return_convergence_delta=False,
        )

        attr_left, attr_center, attr_right, attr_meta = attrs
        abs_meta_attrs.append(attr_meta[0].detach().cpu().abs().numpy())
        _, modality_percent = modality_contributions(attr_left[0], attr_center[0], attr_right[0], attr_meta[0])
        modality_rows.append(modality_percent)

    global_meta = np.mean(np.stack(abs_meta_attrs, axis=0), axis=0)

    if len(feature_names) != len(global_meta):
        raise ValueError("selected metadata feature names must match the final model metadata dimension.")

    global_meta_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_attribution": global_meta,
        }
    ).sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)

    global_modality_df = pd.DataFrame(modality_rows).mean().sort_values(ascending=False).reset_index()
    global_modality_df.columns = ["modality", "mean_percent_contribution"]

    return global_meta_df, global_modality_df
