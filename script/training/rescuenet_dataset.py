from pathlib import Path
import os

import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np

class RescueNetDataset(Dataset):
  def __init__(self, data_dir: str, transform=None):
    super().__init__()
    
    self.transform = transform
    self.data_dir = data_dir
    basename = os.path.basename(self.data_dir.rstrip('/'))
    
    self.image_dir = Path(self.data_dir) / f'{basename}-org-img'
    self.label_dir = Path(self.data_dir) / f'{basename}-label-img'
    self.depth_dir = Path(self.data_dir) / f'{basename}-depth-img'
    self.vis_dir = Path(self.data_dir) / f'{basename}-vis-img'
    print(str(self.depth_dir))
    
    self.image_paths = sorted(list(self.image_dir.glob('*.jpg')))
    self.ids = [p.name.split('.')[0] for p in self.image_paths]
    
    self.label_paths = sorted(list(self.label_dir.glob('*.png')))
    self.depth_paths = sorted(list(self.depth_dir.glob('*.npy')))
    self.vis_paths = sorted(list(self.vis_dir.glob('*.png')))
    
  def __len__(self):
    return len(self.ids)
    
  def __getitem__(self, index):
    image_path = self.image_paths[index]
    label_path = self.label_paths[index]
    depth_path = self.depth_paths[index]
    entry_id = self.ids[index]
    
    image = io.imread(image_path)
    label = io.imread(label_path)
    depth = np.load(depth_path).squeeze()
    
    if self.transform:
      augmented = self.transform(image=image, mask=label, depth=depth)
      image = augmented["image"]
      label = augmented["mask"]
      depth = augmented["depth"]
    
    image_matrix = np.transpose(image, (2, 0, 1))
    image_tensor = torch.from_numpy(image_matrix).float()
    
    if image_tensor.max() > 1.0:
      image_tensor /= 255.0

    label_tensor = torch.from_numpy(label).long()
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
    
    return entry_id, image_tensor, label_tensor, depth_tensor
