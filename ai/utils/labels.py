from typing import List, Tuple, Literal
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2

from custom_types.datasets import YOLOSegmentationTypedDict

def convert_png_to_yolo_label(label: np.ndarray) -> YOLOSegmentationTypedDict:
  yolo_annotations = {
    'class_ids': [],
    'bboxes': [],
    'masks': []
  }
  height, width = label.shape

  unique_classes = np.unique(label)
  unique_classes = unique_classes[unique_classes != 0]
  
  for class_id in unique_classes:
    class_mask = np.where(label == class_id, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
      if cv2.contourArea(cnt) < 10:
        continue
      
      coords = cnt.squeeze()
      if coords.ndim == 1:
        continue
      
      yolo_annotations['class_ids'].append(class_id)
      yolo_annotations['masks'].append(coords)
      
      x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
      
      x = (x_box + (w_box / 2)) / width
      y = (y_box + (h_box / 2)) / height
      w = w_box / width
      h = h_box / height
      
      yolo_annotations['bboxes'].append([x, y, w, h])
  
  return yolo_annotations

def create_binary_mask(coords: List[np.ndarray], dims: Tuple[int, int] = (640, 640)) -> torch.Tensor:
  zeros = np.zeros(dims, dtype=np.uint8)
  
  poly_points = np.int32([coords])
  binary_mask = cv2.fillPoly(zeros, poly_points, 1)
  binary_mask_tensor = torch.from_numpy(binary_mask).float()
  
  return binary_mask_tensor

class CameraIntrinsics(BaseModel):
  model_config = ConfigDict(frozen=True)
  
  fx: PositiveFloat
  fy: PositiveFloat
  cx: float
  cy: float
  width: PositiveInt
  height: PositiveInt
  
  def matrix(self) -> np.ndarray:
    return np.array(
      [
        [self.fx, 0, self.cx],
        [0, self.fy, self.cy],
        [0, 0, 1]
      ]
    )

  def scaled(self, width: int, height: int) -> "CameraIntrinsics":
    sx, sy = width / self.width, height / self.height
    return CameraIntrinsics(fx=self.fx * sx, fy=self.fy * sy,
                            cx=self.cx * sx, cy=self.cy * sy,
                            width=width, height=height)

  def check(self, depth: np.ndarray) -> None:
    if depth.shape != (self.height, self.width):
      raise ValueError(
        f"depth is {depth.shape}, intrinsics are for {(self.height, self.width)}")

  def unproject(self, depth: np.ndarray):
    """
      Unproject indices to coordinates
      z[u, v, 1]^T = Kp_i = [ f_x 0   u_0 ] [ x ]
                            | 0   f_y v_0 | | y |
                            [ 0   0   1   ] [ z ]
      
      z * u = x * f_x + z * u_0
      x * f_x = z * u - z * u_0
      x * f_x = z * (u - u_0)
      x = z * (u - u_0) / f_x
      
      z * v = y * f_y + z * v_0
      y * f_y = z * v - z * v_0
      y * f_y = z * (v - v_0)
      y = z * (v - v_0) / f_y
    """
    self.check(depth)
    u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
    
    x = (u - self.cx) * depth / self.fx
    y = (v - self.cy) * depth / self.fy
    
    return x, y, depth
  
  def project(self, x, y, z):
    u = self.fx * x / z + self.cx
    v = self.fy * y / z + self.cy
    return u, v

def shift_matrix(original_matrix: np.ndarray, shift: int, axis: int) -> np.ndarray:
  shifted_matrix = np.roll(original_matrix, shift=shift, axis=axis)
  
  # Shifting Rows (Up/Down)
  if axis == 0:
    if shift > 0:
      shifted_matrix[:shift, :] = np.inf
    elif shift < 0:
      shifted_matrix[shift:, :] = np.inf
  # Shifting Columns (Left/Right)
  elif axis == 1:
    if shift > 0:
      shifted_matrix[:, :shift] = np.inf
    elif shift < 0:
      shifted_matrix[:, shift:] = np.inf
          
  return shifted_matrix

def select_eta(second_d: np.ndarray, mode: Literal["u", "v"]) -> np.ndarray:
  padding_width = ((0, 0), (1, 1)) if mode == "u" else ((1, 1), (0, 0))
  padded = np.pad(second_d, padding_width, constant_values=np.inf)
  
  window = sliding_window_view(padded, 3, axis=1 if mode == "u" else 0)
  window = window[..., [1, 0, 2]]
  
  eta = np.array([0, -1, 1])[np.argmin(window, axis=-1)]
  
  return eta
  
def sda_sne(depth_map: np.ndarray, camera_intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
  # Initialization Phase
  
  first_dufw_shifted = shift_matrix(depth_map, shift=-1, axis=1)
  first_dubw_shifted = shift_matrix(depth_map, shift=+1, axis=1)
  first_dvfw_shifted = shift_matrix(depth_map, shift=-1, axis=0)
  first_dvbw_shifted = shift_matrix(depth_map, shift=+1, axis=0)
  
  first_dufw = first_dufw_shifted - depth_map
  first_dubw = depth_map - first_dubw_shifted
  first_dvfw = first_dvfw_shifted - depth_map
  first_dvbw = depth_map - first_dvbw_shifted
  second_du = first_dufw_shifted - 2 * depth_map + first_dubw_shifted
  second_dv = first_dvfw_shifted - 2 * depth_map + first_dvbw_shifted
  
  eta_smoothing_u = select_eta(second_du, "u")
  eta_smoothing_v = select_eta(second_dv, "v")
  
  z_u = np.select([eta_smoothing_u == 1, eta_smoothing_u == -1],
                  [first_dufw, first_dubw],
                  default=0.5 * (first_dufw + first_dubw))
  z_v = np.select([eta_smoothing_v == 1, eta_smoothing_v == -1],
                  [first_dvfw, first_dvbw],
                  default=0.5 * (first_dvfw + first_dvbw))
  
  # Iteration Phase
  
  return dz_du, dz_dv

def compute_nz(
  x: np.ndarray,
  y: np.ndarray,
  z: np.ndarray,
  n_x: np.ndarray,
  n_z: np.ndarray,
  window_size: int
) -> np.ndarray:
  n_z = np.array()
  
  return n_z

def generate_surface_normals(
  depth_map: np.ndarray,
  camera_intrinsics: CameraIntrinsics,
  window_size: int = 3
) -> np.ndarray:
  """
    Unproject to get the x, y, z coordinates.
  """
  x, y, z = camera_intrinsics.unproject(depth_map)
  
  """
    Compute the depth gradients of the whole image
    - Shape = (width, height)
  """
  dz_du, dz_dv = sda_sne(depth_map, camera_intrinsics)
  inv_dz_du, inv_dz_dv = -dz_du ** 2, - dz_dv ** 2
  
  """
    Compute the normal components
    - n_x = intrinsics x * inverse depth graident in respect to u - Shape = (width, height).
    - n_y = intrinsics y * inverse depth graident in respect to v - Shape = (width, height).
    - n_z (follows the compute function)
  """
  n_x = camera_intrinsics.fx * inv_dz_du
  n_y = camera_intrinsics.fy * inv_dz_dv
  n_z = compute_nz(x, y, z, n_y, n_z, window_size)
  
  surface_normals = np.stack([n_x, n_y, n_z], axis=-1)
  surface_normals /= np.max(np.linalg.norm(surface_normals, axis=-1, keepdims=True), 1e-12)
  
  p = np.stack([x, y, z], axis=-1)
  surface_normals[(surface_normals * p).sum(-1) > 0] *= -1                               # face the camera
  return surface_normals