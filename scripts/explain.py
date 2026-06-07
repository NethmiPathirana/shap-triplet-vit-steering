from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from triplet_steering.config import load_config
from triplet_steering.pipeline import build_loaders_for_selected_data, prepare_selected_data_from_checkpoint
from triplet_steering.training.checkpoints import build_model_from_checkpoint, load_checkpoint
from triplet_steering.utils import get_device, save_dataframe_csv, set_seed
from triplet_steering.xai.captum_explain import explain_multiple_samples, global_metadata_xai


def parse_args():
    parser = argparse.ArgumentParser(description="Run Captum explainability for a trained triplet multimodal model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample-indices", type=int, nargs="*", default=None)
    parser.add_argument("--random-samples", action="store_true")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--global-xai", action="store_true")
    parser.add_argument("--global-samples", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=24)
    parser.add_argument("--nt-samples", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device()

    checkpoint = load_checkpoint(args.checkpoint, device=device)
    selected_data = prepare_selected_data_from_checkpoint(cfg, checkpoint)
    _, val_dataset, _, _ = build_loaders_for_selected_data(selected_data, cfg)

    model = build_model_from_checkpoint(cfg, checkpoint, device=device, pretrained=False)
    output_dir = Path(cfg.checkpoints_dir) / "xai"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_indices is not None and len(args.sample_indices) > 0:
        sample_indices = args.sample_indices[: args.num_samples]
    elif args.random_samples:
        rng = np.random.default_rng(cfg.seed)
        sample_indices = rng.choice(len(val_dataset), size=min(args.num_samples, len(val_dataset)), replace=False).tolist()
    else:
        sample_indices = list(range(min(args.num_samples, len(val_dataset))))

    summary_df = explain_multiple_samples(
        model=model,
        dataset=val_dataset,
        sample_indices=sample_indices,
        feature_names=selected_data.selected_features,
        device=device,
        output_dir=output_dir,
        use_noise_tunnel=True,
        nt_samples=args.nt_samples,
        n_steps=args.n_steps,
    )

    save_dataframe_csv(summary_df, output_dir / "xai_summary.csv")

    if args.global_xai:
        global_meta_df, global_modality_df = global_metadata_xai(
            model=model,
            dataset=val_dataset,
            feature_names=selected_data.selected_features,
            device=device,
            n_samples=args.global_samples,
            n_steps=args.n_steps,
            seed=cfg.seed,
        )
        save_dataframe_csv(global_meta_df, output_dir / "global_metadata_xai.csv")
        save_dataframe_csv(global_modality_df, output_dir / "global_modality_xai.csv")

    print("Explainability completed.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
