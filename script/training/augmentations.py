import albumentations as A
import cv2

import numpy as np

error = (1/3) * np.sum(np.square(np.array([0, -1, -2])))

def get_augmentation_pipeline(
  height: float=640.0,
  width: float=640.0
) -> A.Compose:
  return A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(
      scale=(0, 0.15),
      interpolation=cv2.INTER_LINEAR,
      mask_interpolation=cv2.INTER_NEAREST,
      p=0.5
    ),
    A.OneOf([
      A.MotionBlur(p=0.5),
      A.GaussNoise(),
    ], p=0.3),
    A.RandomBrightnessContrast(
      brightness_range=(0.0, 0.2),
      contrast_range=(0.0, 0.2),
      p=0.4
    ),
    A.HueSaturationValue(),
    A.CoarseDropout(
      num_holes_range=(1, 8),
      fill_mask=0,
      p=0.2
    ),
    A.Resize(
      height=height,
      width=width,
      interpolation=cv2.INTER_CUBIC,
      mask_interpolation=cv2.INTER_NEAREST
    ),
  ])