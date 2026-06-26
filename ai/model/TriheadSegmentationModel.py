from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from .FeatureDecoderHead import DecoderHead
from .DisasterSiteMap import DisasterSiteMap

class TriheadSegmentationModel(torch.nn.Module):
  intercepted_features: Optional[List[torch.Tensor]] = [None] * 3
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
    
    self.feature_size = 256
    self.yolo_pt_path = yolo_pt_path
    self.device = device
    self.verbose = verbose
    self.extrinsics = extrinsics
    self.intrinsics = intrinsics
    self.include_depth = include_depth
    self.include_normals = include_normals
    
    yolo_wrapper = YOLO(model=yolo_pt_path, verbose=verbose)
    
    self.yolo_backbone = yolo_wrapper.model.to(device)
    
    if self.include_depth or self.include_normals:
      def intercept_feature_map(_module, _input, output, store_idx):
        if isinstance(output, tuple) or isinstance(output, list):
          self.intercepted_features[store_idx] = output[-1].to(device=device) 
        else:
          self.intercepted_features[store_idx] = output.to(device=device)

      for i, module_idx in enumerate([4, 6, 10]):
        self.yolo_backbone.model[module_idx].register_forward_hook(lambda _module, _input, output, idx=i: intercept_feature_map(_module, _input, output, idx))
    
    if self.include_depth:
      self.depth_head = DecoderHead(out_channels=1, target_size=self.target_size).to(device)
    
    if self.include_normals:
      self.normal_head = torch.nn.Sequential(
        DecoderHead(out_channels=3, target_size=self.target_size),
        torch.nn.Tanh()
      ).to(device)
  
  def stitch_maps(self, maps_list: List[np.ndarray]) -> np.ndarray:
    return np.array([])
  
  def project_point_cloud_to_2d(self) -> np.ndarray:
    return np.array([])
  
  def forward(self, X: torch.Tensor, *args, **kwds):
    segmentation_map = self.yolo_backbone._predict_once(X)
    
    depth_out = normal_out = None
    
    p5, p4, p3 = self.intercepted_features[2], self.intercepted_features[1], self.intercepted_features[0]
    if self.include_depth:
      depth_out = self.depth_head(p5, p4, p3)
    if self.include_normals:
      normal_out = self.normal_head(p5, p4, p3)
    
    return segmentation_map, depth_out, normal_out