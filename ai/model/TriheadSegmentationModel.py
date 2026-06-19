from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from .DisasterSiteMap import DisasterSiteMap

class TriheadSegmentationModel(torch.nn.Module):
  yolo_pt_path: str
  yolo: YOLO
  intercepted_features: Optional[torch.Tensor] = None
  extrinsics: Optional[np.ndarray] = None
  intrinsics: Optional[np.ndarray] = None
  images: List[np.ndarray] = []
  stitched_map: DisasterSiteMap = DisasterSiteMap()
  target_size: Tuple[int, int] = (640, 640)
  device: str
  in_channels: int
  
  def __init__(
    self,
    yolo_pt_path: str,
    device: str,
    include_depth: bool = True,
    include_normals: bool = True,
    verbose: bool = False,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
  ):
    super(TriheadSegmentationModel, self).__init__()
    
    self.feature_size = 512
    self.yolo_pt_path = yolo_pt_path
    self.device = device
    self.verbose = verbose
    self.extrinsics = extrinsics
    self.intrinsics = intrinsics
    self.include_depth = include_depth
    self.include_normals = include_normals
    
    self.yolo = YOLO(model=yolo_pt_path, task='segment', verbose=verbose).to(device)
    self.yolo.model.to(device)
    
    if self.include_depth or self.include_normals:
      def intercept_feature_map(_module, _input, output):
        self.intercepted_features = output.to(device=device)
      self.yolo.model.model[22].register_forward_hook(intercept_feature_map)
    
    if self.include_depth:
      self.depth_head = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=self.feature_size, out_channels=256, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256, 1, kernel_size=1),
      ).to(device)
    
    if self.include_normals:
      self.normal_head = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=self.feature_size, out_channels=256, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256, 3, kernel_size=1),
        torch.nn.Tanh(),
      ).to(device)
  
  def stitch_maps(self, maps_list: List[np.ndarray]) -> np.ndarray:
    return np.array([])
  
  def project_point_cloud_to_2d(self) -> np.ndarray:
    return np.array([])
  
  def forward(self, X: torch.Tensor, *args, **kwds):
    segmentation_map = self.yolo.model(X)
    
    if self.include_depth:
      depth_out = self.depth_head(self.intercepted_features)
      depth_out = F.interpolate(depth_out, size=self.target_size, mode='bilinear')
    else:
      depth_out = None
    
    if self.include_normals:
      normal_out = self.normal_head(self.intercepted_features)
      normal_out = F.interpolate(normal_out, size=self.target_size, mode='bilinear')
    else:
      normal_out = None
    
    return segmentation_map, depth_out, normal_out