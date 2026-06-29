from pathlib import Path
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage import io
import numpy as np

def collate_fn(batch):
  images, targets = zip(*batch)
  include_depth, include_normals = targets[0]['include_depth'], targets[0]['include_normals']
  
  batch_images = torch.stack(images, dim=0)
  
  batch_depth, batch_normals = [], []
  batch_sem_masks = []
  
  for i, target in enumerate(targets):
    batch_sem_masks.append(target['semantic_mask'])
    batch_depth.append(target['depth_map'] if include_depth else [])
    batch_normals.append(target['surface_normals'] if include_normals else [])
    
  return batch_images, (
    {
      'semantic_mask': torch.stack(batch_sem_masks, dim=0),
    },
    {
      'depth': torch.stack([d.squeeze() for d in batch_depth], dim=0).unsqueeze(1),
    } if include_depth else None,
    {
      'normals': torch.stack([n.squeeze() for n in batch_normals], dim=0)
    } if include_normals else None
  )

class RescueNetDataset(Dataset):
  def __init__(
    self,
    data_dir: str,
    training: bool = False,
    include_depth: bool = True,
    include_normals: bool = True,
    spatial_transform=None,
    photometric_transform=None
  ):
    super().__init__()
    
    self.include_depth = include_depth
    self.include_normals = include_normals
    self.training = training
    
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
    target = {}
    target['include_depth'] = self.include_depth
    target['include_normals'] = self.include_normals
    
    image_path = self.image_paths[index]
    label_path = self.label_paths[index]
    depth_path = self.depth_paths[index] if self.include_depth else None
    normals_path = self.normals_paths[index] if self.include_normals else None
    
    image = io.imread(image_path)
    
    label = np.load(label_path).squeeze()
    
    depth = None
    if self.include_depth:
      depth = np.load(depth_path).squeeze()
      depth = depth.astype(np.float32)

    normals = None
    if self.include_normals:
      normals = np.load(normals_path).squeeze()
      normals = np.transpose(normals, [1, 2, 0])
      normals = normals.astype(np.float32)
    
    if not self.training:
      target['semantic_mask'] = torch.from_numpy(label).long()
      
      if self.include_depth:
        target['depth_map'] = torch.from_numpy(depth).unsqueeze(0)
      
      if self.include_normals:
        normals_tensor = torch.from_numpy(normals).permute(2, 0, 1).float()
        target['surface_normals'] = normals_tensor
      
      val_image = torch.from_numpy(image).permute(2, 0, 1).float()
      if val_image.max() > 1.0:
          val_image /= 255.0
      
      return val_image, target
    
    aug_rgb = image.copy()
    aug_mask = label
    aug_depth = depth
    aug_normals = normals
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
            aug_normals[0, :, :] *= -1.0 
              
          if transform['__class_fullname__'] == 'VerticalFlip' and transform['applied']:
            aug_normals[1, :, :] *= -1.0
          
          if transform['__class_fullname__'] == 'SafeRotate' and transform['applied']:
            transform_params = transform['params']

            rotate = transform_params['rotate']
            angle_rad = np.deg2rad(rotate)
            sin_theta, cos_theta = np.sin(angle_rad), np.cos(angle_rad)

            nx = aug_normals[0, :, :].clone()
            ny = aug_normals[1, :, :].clone()
            
            aug_normals[0, :, :] = cos_theta * nx - sin_theta * ny
            aug_normals[1, :, :] = sin_theta * nx + cos_theta * ny

        aug_normals = F.normalize(aug_normals.float(), dim=0)
    
    if self.photometric_transform:
      aug_rgb = self.photometric_transform(image=aug_rgb.permute(1, 2, 0).numpy())['image']
    
    aug_rgb = aug_rgb.float()
    if aug_rgb.max() > 1.0:
      aug_rgb /= 255.0

    label_tensor = torch.as_tensor(aug_mask, dtype=torch.long)
    target['semantic_mask'] = label_tensor
    
    if self.include_depth:
      if isinstance(aug_depth, np.ndarray):
        depth_tensor = torch.from_numpy(aug_depth).unsqueeze(0).float()
      else:
        depth_tensor = aug_depth.float()
      target['depth_map'] = depth_tensor
    
    if self.include_normals:
      target['surface_normals'] = aug_normals
    
    return aug_rgb, target
