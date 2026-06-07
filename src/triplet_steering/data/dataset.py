from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class TripletDrivingDataset(Dataset):
    def __init__(self, dataframe, metadata: np.ndarray, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.metadata = metadata.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]

        left_image = Image.open(row["left_path"]).convert("RGB")
        center_image = Image.open(row["center_path"]).convert("RGB")
        right_image = Image.open(row["right_path"]).convert("RGB")

        target = float(row["steering"])
        left_image, center_image, right_image, target = self.transform(left_image, center_image, right_image, target)

        return {
            "left": left_image,
            "center": center_image,
            "right": right_image,
            "meta": torch.tensor(self.metadata[index], dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "left_path": row["left_path"],
            "center_path": row["center_path"],
            "right_path": row["right_path"],
        }
