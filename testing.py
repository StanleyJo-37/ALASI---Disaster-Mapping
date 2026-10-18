import os
from typing import Optional

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from ai.utils.labels import generate_surface_normals, CameraIntrinsics

# ds = load_dataset("sayakpaul/nyu_depth_v2", revision="refs/convert/parquet", split="train", streaming=True)

# os.makedirs('test_data', exist_ok=True)

# for i, ex in enumerate(ds.take(20)):
#   print(ex.keys())
#   ex["image"].save(f"test_data/rgb_{i}.png")
  
#   depth = np.array(ex["depth_map"])
#   np.save(f'test_data/depth_{i}.npy', depth)

def color_depthmap(depth_map: np.ndarray, d_min: Optional[int], d_max: Optional[int]) -> np.ndarray:
  if d_min is None:
    d_min = np.min(depth_map)
  if d_max is None:
    d_max = np.max(depth_map)
  
  normalized_depth_map = (depth_map - d_min) / (d_max - d_min)
  return 255 * plt.cm.viridis(normalized_depth_map)[:, :, :3]

def show_results(depth_map: np.ndarray, surface_normals):
  if depth_map.ndim == 3:
    depth_map = depth_map.squeeze()
    
  colored_depthmap = color_depthmap(depth_map, None, None)
  
  # plt.subplot(1, 2, 1)
  # plt.imshow(colored_depthmap.astype(np.uint8))
  # plt.axis('off')
  
  # plt.subplot(1, 2, 2)
  normals_vis = np.clip((1 - surface_normals) / 2, 0, 1)
  # normals_vis[~(depth_map < 1.0)] = 1
  plt.imshow(normals_vis)
  plt.axis('off')
  plt.show()
  
cameraInstrinsics = CameraIntrinsics(
  # fx=518.8579011745019,
  fx=525.0,
  # fy=519.4696111212749,
  fy=525.0,
  # cx=325.5824494111903,
  cx=320.0,
  # cy=253.73616633400465,
  cy=240.0,
  width=640,
  height=480
)
  
def fill_missing_depth(depth_map: np.ndarray) -> np.ndarray:
  fill_mask = (depth_map == 0)
  _, nearest_indices = distance_transform_edt(fill_mask, return_indices=True)
  
  return depth_map[tuple(nearest_indices)]
  
for i in range(1):
  # depth_map = np.load(f'test_data/depth_{i}.npy')
  depth_map = np.load(f'test_data/10778.npy')
  # raw_data = np.fromfile('test_data/000001.bin', dtype=np.float32)
  # depth_map = raw_data.reshape((480, 640))
  clean_depth_map = fill_missing_depth(depth_map)
  surface_normals = generate_surface_normals(clean_depth_map, cameraInstrinsics)
  
  show_results(clean_depth_map, surface_normals)