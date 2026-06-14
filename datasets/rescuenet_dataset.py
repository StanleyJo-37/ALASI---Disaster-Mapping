from pathlib import Path
import os

import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np

class RescueNetDataset(Dataset):
  def __init__(
    self,
    data_dir: str,
    include_depth: bool = True,
    include_normals: bool = True,
    spatial_transform=None,
    photometric_transform=None
  ):
    super().__init__()
    
    self.include_depth = include_depth
    self.include_normals = include_normals
    
    self.spatial_transform = spatial_transform
    self.photometric_transform = photometric_transform
    
    self.data_dir = data_dir
    basename = os.path.basename(self.data_dir.rstrip('/'))
    
    self.image_dir = Path(self.data_dir) / f'{basename}-resized-img'
    self.label_dir = Path(self.data_dir) / f'{basename}-labelnpy-img'
    self.depth_dir = Path(self.data_dir) / f'{basename}-depth-img' if self.include_depth else None
    self.normals_dir = Path(self.data_dir) / f'{basename}-normal-img' if self.include_normals else None
    
    self.image_paths = sorted(list(self.image_dir.glob('*.jpg')))
    self.ids = [p.name.split('.')[0] for p in self.image_paths]
    
    self.label_paths = sorted(list(self.label_dir.glob('*.npy')))
    self.depth_paths = sorted(list(self.depth_dir.glob('*.npy'))) if self.include_depth else []
    self.normals_paths = sorted(list(self.normals_dir.glob('*.npy'))) if self.include_normals else []
    
  def __len__(self):
    return len(self.ids)
    
  def __getitem__(self, index):
    image_path = self.image_paths[index]
    label_path = self.label_paths[index]
    depth_path = self.depth_paths[index] if self.include_depth else None
    normals_path = self.normals_paths[index] if self.include_normals else None
    
    image = io.imread(image_path)
    
    label = np.load(label_path).squeeze()
    
    if self.include_depth:
      depth = np.load(depth_path).squeeze()
      depth = depth.astype(np.float32)

    if self.include_normals:
      normals = np.load(normals_path).squeeze()
      normals = np.transpose(normals, [1, 2, 0])
      normals = normals.astype(np.float32)
    
    aug_rgb = image.copy()
    if self.spatial_transform:
      augmented = self.spatial_transform(
        image=image,
        mask=label,
        depth=depth if self.include_depth else None,
        normals=normals if self.include_normals else None
      )
      
      aug_rgb = augmented['image']
      aug_mask = augmented['mask']
      aug_depth = augmented['depth'] if self.include_depth else None
      aug_normals = augmented['normals'] if self.include_normals else None
      
      if self.include_normals:
        for transform in augmented['replay']['transforms']:
          if transform['__class_fullname__'] == 'HorizontalFlip' and transform['applied']:
            aug_normals[:, :, 0] *= -1.0 
              
          if transform['__class_fullname__'] == 'VerticalFlip' and transform['applied']:
            aug_normals[:, :, 1] *= -1.0
      
        norm = np.linalg.norm(aug_normals, axis=-1, keepdims=True)
        aug_normals = np.divide(aug_normals, norm, out=np.zeros_like(aug_normals), where=norm!=0)
    
    if self.photometric_transform:
      aug_rgb = self.photometric_transform(image=aug_rgb.permute(1, 2, 0).numpy())['image']
    
    aug_rgb = aug_rgb.float()
    if aug_rgb.max() > 1.0:
      aug_rgb /= 255.0

    label_tensor = aug_mask.long()
    depth_tensor = aug_depth.unsqueeze(0).float() if self.include_depth else None
    
    target = {
      'segmentation': label_tensor
    }
    
    if self.include_depth:
      target['depth_map'] = depth_tensor
    
    if self.include_normals:
      target['surface_normals'] = aug_normals
    
    return aug_rgb, target
