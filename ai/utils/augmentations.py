from typing import Tuple

import albumentations as A
import cv2

def get_augmentation_pipeline() -> Tuple[A.Compose, A.Compose]:
  spatial = A.ReplayCompose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.SafeRotate(
      angle_range=(-180, 180),
      interpolation=cv2.INTER_LINEAR, 
      mask_interpolation=cv2.INTER_NEAREST,
      p=0.5
    ),
    A.Affine(
      scale=(0.85, 1.15),
      interpolation=cv2.INTER_LINEAR,
      mask_interpolation=cv2.INTER_NEAREST,
      p=0.5
    ),
    A.ToTensorV2()
  ], additional_targets={'depth': 'image', 'normals': 'image'})
  
  photometric = A.Compose([
    A.OneOf([
      A.MotionBlur(p=0.5),
      A.GaussNoise(),
    ], p=0.3),
    A.RandomBrightnessContrast(
      brightness_range=(-0.2, 0.2),
      contrast_range=(-0.2, 0.2),
      p=0.4
    ),
    A.HueSaturationValue(p=0.3),
    A.CoarseDropout(
      num_holes_range=(1, 8),
      fill_mask=0,
      p=0.2
    ),
    A.ToTensorV2()
  ])
  
  return spatial, photometric