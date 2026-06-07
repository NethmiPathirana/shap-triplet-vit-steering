from __future__ import annotations

import random

import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TripletTransform:
    def __init__(self, img_size: int = 224, train: bool = True):
        self.img_size = img_size
        self.train = train

    def _resize(self, image):
        target = self.img_size + 32 if self.train else self.img_size
        return TF.resize(image, [target, target], interpolation=InterpolationMode.BILINEAR)

    def _random_crop_params(self, image):
        width, height = image.size
        target_height, target_width = self.img_size, self.img_size
        if height == target_height and width == target_width:
            return 0, 0, height, width
        top = random.randint(0, height - target_height)
        left = random.randint(0, width - target_width)
        return top, left, target_height, target_width

    def _center_crop_params(self, image):
        width, height = image.size
        target_height, target_width = self.img_size, self.img_size
        top = max((height - target_height) // 2, 0)
        left = max((width - target_width) // 2, 0)
        return top, left, target_height, target_width

    @staticmethod
    def _crop(image, params):
        top, left, height, width = params
        return TF.crop(image, top, left, height, width)

    @staticmethod
    def _normalize(image):
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, left_image, center_image, right_image, target: float):
        left_image = self._resize(left_image)
        center_image = self._resize(center_image)
        right_image = self._resize(right_image)

        params = self._random_crop_params(center_image) if self.train else self._center_crop_params(center_image)

        left_image = self._crop(left_image, params)
        center_image = self._crop(center_image, params)
        right_image = self._crop(right_image, params)

        if self.train and random.random() < 0.5:
            left_image, right_image = right_image, left_image
            left_image = TF.hflip(left_image)
            center_image = TF.hflip(center_image)
            right_image = TF.hflip(right_image)
            target = -target

        left_tensor = self._normalize(left_image)
        center_tensor = self._normalize(center_image)
        right_tensor = self._normalize(right_image)

        return left_tensor, center_tensor, right_tensor, float(np.clip(target, -1.0, 1.0))
