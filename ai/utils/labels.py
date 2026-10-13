from typing import TypeAlias, List, Tuple, Literal, Optional, Dict
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
import warnings
import re

from PIL import ImageFile
import xml.etree.ElementTree as ET
import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2

from ..custom_types.datasets import YOLOSegmentationTypedDict

AxisType: TypeAlias = Literal["u", "v"]

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

def get_camera_intrinsics(img: ImageFile.ImageFile) -> CameraIntrinsics:
  DEFAULT_FX = 3051.613037
  DEFAULT_FY = 3051.613037
  DEFAULT_CX = 2500
  DEFAULT_CY = 2500
  
  focal_length_regex = r'drone-dji:CalibratedFocalLength="([\d.]+)"'
  center_x_regex = r'drone-dji:CalibratedOpticalCenterX="([\d.]+)"'
  center_y_regex = r'drone-dji:CalibratedOpticalCenterY="([\d.]+)"'

  xmp_data = img.info.get("xmp")
  width, height = img.size
  
  if not xmp_data:
    print("No XMP calibration data found in this image.")
    return CameraIntrinsics(
      fx=DEFAULT_FX,
      fy=DEFAULT_FY,
      cx=DEFAULT_CX,
      cy=DEFAULT_CY,
      width=width,
      height=height
    )

  xmp_str = xmp_data.decode('utf-8', errors='ignore') if isinstance(xmp_data, bytes) else xmp_data
  
  focal_length = float(re.findall(focal_length_regex, xmp_str)[0])
  center_x = float(re.findall(center_x_regex, xmp_str)[0])
  center_y = float(re.findall(center_y_regex, xmp_str)[0])
  
  if np.isnan(focal_length) or np.isnan(center_x) or np.isnan(center_y):
    return CameraIntrinsics(
      fx=np.where(np.isnan(focal_length), DEFAULT_FX, focal_length),
      fy=np.where(np.isnan(focal_length), DEFAULT_FY, focal_length),
      cx=np.where(np.isnan(center_x), DEFAULT_CX, center_x),
      cy=np.where(np.isnan(center_y), DEFAULT_CY, center_y),
      width=width,
      height=height
    )
  
  return CameraIntrinsics(
    fx=focal_length,
    fy=focal_length,
    cx=center_x,
    cy=center_y,
    width=width,
    height=height
  )

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

def initialize_eta_and_energy(second_d: np.ndarray, mode: AxisType) -> np.ndarray:
  padding_width = ((0, 0), (1, 1)) if mode == "u" else ((1, 1), (0, 0))
  padded = np.pad(np.abs(second_d), padding_width, constant_values=np.inf)
  
  window = sliding_window_view(padded, 3, axis=1 if mode == "u" else 0)
  window = window[..., [1, 0, 2]]
  
  state = np.array([0, -1, 1], dtype=np.int8)[np.argmin(window, axis=-1)]
  energy = window.min(axis=-1)
  
  return state, energy
  
OFFSETS = np.array([(0,0), (-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,-1), (-1,1), (1,1)])  # [du, dv]
def select_state(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  best = omega.argmin(axis=0)
  state = OFFSETS[best]
  energy = np.take_along_axis(omega, best[None, ...], axis=0)[0]
  return state, energy
  
def nb(
  a: np.ndarray,
  du: np.ndarray,
  dv: np.ndarray,
  fill: np.any=np.inf
):
  H, W = a.shape
  out = np.full(a.shape, fill, dtype=a.dtype)
  out[max(-dv,0):H-max(dv,0), max(-du,0):W-max(du,0)] = \
      a[max(dv,0):H+min(dv,0), max(du,0):W+min(du,0)]
  return out
  
INDICATOR_INDICES: Dict[AxisType, Dict[Literal[-1, 1], int]] = {
  "u": {
    -1: 0,
    1: 1,
  },
  "v": {
    -1: 2,
    1: 3
  }
}
  
def build_omega(
  E_p: np.ndarray,
  E_o: np.ndarray,
  a_pp: np.ndarray,
  a_oo: np.ndarray,
  channel: AxisType,
  indicators: List[np.ndarray]
):
  cands = []
  for du, dv in OFFSETS:
    along, across = (du, dv) if channel == "u" else (dv, du)
    
    if along == 0 and across == 0:
      c = E_p
    elif along == 0:
      c = 2 * (nb(a_oo, du, dv) + nb(E_p, du, dv))
    elif across == 0:
      c = nb(a_pp, du, dv) + indicators[INDICATOR_INDICES[channel][along]]
    else:
      odu, odv = (0, dv) if channel == "u" else (du, 0)
      c = nb(a_oo, odu, odv) + nb(E_p, odu, odv) + nb(a_pp, du, dv) + nb(E_o, du, dv)
    
    cands.append(c)
    
  return np.stack(cands, axis=0)
  
def build_gradient_candidates(
  z: np.ndarray,
  G_p: np.ndarray,
  G_o: np.ndarray,
  channel: AxisType,
  k: int
):
  cands = []
  for du, dv in OFFSETS:
    along, across = (du, dv) if channel == "u" else (dv, du)
    if along == 0 and across == 0:
      c = G_p
    elif along == 0:
      c = nb(G_p, du, dv)
    elif across == 0:
      c = G_p + along * ((nb(z, du, dv) - z) / (k + 1)) - (nb(G_p, du, dv) / (k + 1))
    else:
      odu, odv = (0, dv) if channel == "u" else (du, 0) 
      c = nb(G_p, odu, odv) + du * dv * (G_o - nb(G_o, du, dv))
    cands.append(c)
  
  return np.stack(cands, axis=0)
  
def build_indicator(
  signed_pp: np.ndarray,
  channel: AxisType,
  d: int,
  state: Optional[np.ndarray],
):
  du, dv = (d, 0) if channel == "u" else (0, d)
  
  signed_condition1 = np.where(np.isfinite(nb(signed_pp, du, dv)), nb(signed_pp, du, dv), 0)
  signed_condition2 = np.where(np.isfinite(nb(signed_pp, du*2, dv*2)), nb(signed_pp, du*2, dv*2), 0)
  signed_condition = ~(signed_condition1 * signed_condition2 > 0)
  
  states_directions = (state == np.array([du, dv])).all(axis=-1)
  state_condition = ~(states_directions & nb(states_directions, du, dv, fill=False))
  
  ind_matrix = np.zeros(signed_pp.shape)
  ind_matrix[signed_condition | state_condition] = np.inf
  
  return ind_matrix

def sda(depth_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  H, W = depth_map.shape
  
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
  a_uu, a_vv = np.abs(second_du), np.abs(second_dv)
  
  eta_u, E_u = initialize_eta_and_energy(second_du, "u")
  eta_v, E_v = initialize_eta_and_energy(second_dv, "v")
  
  s_u = np.zeros((H, W, 2), dtype=np.int8); s_u[..., 0] = eta_u
  s_v = np.zeros((H, W, 2), dtype=np.int8); s_v[..., 1] = eta_v
  
  indicators = [
    build_indicator(second_du, "u", -1, s_u),
    build_indicator(second_du, "u",  1, s_u),
    build_indicator(second_dv, "v", -1, s_v),
    build_indicator(second_dv, "v",  1, s_v)
  ]
  
  with np.errstate(invalid="ignore"):
    z_u = np.select([eta_u == 1, eta_u == -1],
                    [first_dufw, first_dubw],
                    default=0.5 * (first_dufw + first_dubw))
    z_v = np.select([eta_v == 1, eta_v == -1],
                    [first_dvfw, first_dvbw],
                    default=0.5 * (first_dvfw + first_dvbw))
  
  # Iteration Phase
  MAX_ITER = int(100)
  for k in range(1, MAX_ITER+1):    
    candidate_u = build_omega(E_u, E_v, a_uu, a_vv, "u", indicators)
    candidate_v = build_omega(E_v, E_u, a_vv, a_uu, "v", indicators)
    
    s_u, E_u = select_state(candidate_u)
    s_v, E_v = select_state(candidate_v)
    
    indicators = [
      build_indicator(second_du, "u", -1, s_u),
      build_indicator(second_du, "u",  1, s_u),
      build_indicator(second_dv, "v", -1, s_v),
      build_indicator(second_dv, "v",  1, s_v)
    ]
    
    if not s_u.any() and not s_v.any():
      break
    
    z_u_cand = build_gradient_candidates(depth_map, z_u, z_v, "u", k)
    z_v_cand = build_gradient_candidates(depth_map, z_v, z_u, "v", k)
    z_u = np.select([(s_u == OFFSETS[m]).all(axis=-1) for m in range(9)], list(z_u_cand))
    z_v = np.select([(s_v == OFFSETS[m]).all(axis=-1) for m in range(9)], list(z_v_cand))
  
  return z_u, z_v

def compute_nz(
  x: np.ndarray,
  y: np.ndarray,
  z: np.ndarray,
  n_x: np.ndarray,
  n_y: np.ndarray,
  window_size: int,
  use_median: bool
) -> np.ndarray:
  r = window_size // 2
  H, W = z.shape
  
  def windows(m):
    padded_m = np.pad(m, ((r, r), (r, r)), constant_values=np.nan)
    return sliding_window_view(padded_m, (window_size, window_size), axis=(0, 1))
  
  dx = windows(x) - x[..., None, None]
  dy = windows(y) - y[..., None, None]
  dz = windows(z) - z[..., None, None]
  dz = np.where(dz == 0, np.nan, dz)
  
  candidates = (dx * n_x[..., None, None] + dy * n_y[..., None, None]) / dz
  candidates = candidates.reshape(H, W, -1)
  
  reduce = np.nanmedian if use_median else np.nanmean
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    phi = reduce(candidates, axis=-1)

  return -phi

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
  d_invz_du, d_invz_dv = sda(1 / depth_map)
  
  """
    Compute the normal components
    - n_x = intrinsics x * inverse depth graident in respect to u - Shape = (width, height).
    - n_y = intrinsics y * inverse depth graident in respect to v - Shape = (width, height).
    - n_z (follows the compute function)
  """
  n_x = camera_intrinsics.fx * d_invz_du
  n_y = camera_intrinsics.fy * d_invz_dv
  n_z = compute_nz(x, y, z, n_x, n_y, window_size=window_size, use_median=True)
  
  surface_normals = np.stack([n_x, n_y, n_z], axis=-1)
  surface_normals /= np.maximum(np.linalg.norm(surface_normals, axis=-1, keepdims=True), 1e-12)
  
  p = np.stack([x, y, z], axis=-1)
  surface_normals[(surface_normals * p).sum(-1) > 0] *= -1
  return surface_normals