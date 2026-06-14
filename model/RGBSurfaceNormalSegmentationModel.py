from typing_extensions import Optional, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from .DisasterSiteMap import DisasterSiteMap

class RGBSurfaceNormalSegmentationModel(torch.nn.Module):
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
    verbose: bool = False,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
  ):
    super(RGBSurfaceNormalSegmentationModel, self).__init__()
    
    self.feature_size = 256
    self.yolo_pt_path = yolo_pt_path
    self.device = device
    self.verbose = verbose
    self.extrinsics = extrinsics
    self.intrinsics = intrinsics
    
    self.yolo = YOLO(model=yolo_pt_path, task='segment', verbose=verbose).to(device)
    
    def intercept_feature_map(_module, _input, output):
      self.intercepted_feature = output
    self.yolo.model.model[22].register_forward_hook(intercept_feature_map)
    
    self.normal_head = torch.nn.Sequential(
      torch.nn.Conv2d(self.feature_size, 128, kernel_size=3, padding=1),
      torch.nn.ReLU(),
      torch.nn.Conv2d(128, 1, kernel_size=1),
      torch.nn.Tanh(),
    )
  
  def stitch_maps(self, maps_list: List[np.ndarray]) -> np.ndarray:
    return np.array([])
  
  def project_point_cloud_to_2d(self) -> np.ndarray:
    return np.array([])
  
  def forward(self, X: torch.Tensor, *args, **kwds):
    segmentation_map = self.yolo(X)
    
    f_map = self.intercepted_features
    
    normal_out = self.normal_head(f_map)
    
    normal_out = F.interpolate(normal_out, size=self.target_size, mode='bilinear')
    
    return segmentation_map, normal_out
