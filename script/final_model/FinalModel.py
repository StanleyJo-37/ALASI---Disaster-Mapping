from typing_extensions import Any, List

import numpy as np
import torch
from ultralytics import YOLO

from lib.depth_anything_3.api import DepthAnything3

class Model():
  yolo_pt_path: str
  da3_model: str
  yolo: YOLO
  da3: DepthAnything3
  extrinsics: np.ndarray[np._AnyShape, np.dtype[Any]] | None = None
  intrinsics: np.ndarray[np._AnyShape, np.dtype[Any]] | None = None
  
  def __init__(
    self,
    yolo_pt_path: str,
    da3_model: str,
    extrinsics: np.ndarray[np._AnyShape, np.dtype[Any]] | None = None,
    intrinsics: np.ndarray[np._AnyShape, np.dtype[Any]] | None = None,
  ):
    self.yolo_pt_path = yolo_pt_path
    self.da3_model = da3_model
    
    self.yolo = YOLO(yolo_pt_path, 'segment', False)
    self.da3 = DepthAnything3.from_pretrained(da3_model)
    
    self.extrinsics = extrinsics
    self.intrinsics = intrinsics
    
  async def generate_depth_map(self, imgs: torch.Tensor) -> np.ndarray[np._AnyShape, np.dtype[Any]]:
    if imgs.shape[0] < 4:
      imgs = imgs.unsqueeze(0)
    
    imgs_np = [np.array(img) for img in imgs.tolist()]
    
    raw_depth_map = self.da3.inference(
      image=imgs_np,
      extrinsics=self.extrinsics,
      intrinsics=self.intrinsics
    )
    
    return raw_depth_map.depth
  
  def stitch_images(images_list: List[np.ndarray]) -> np.ndarray:
    return np.array([])
  
  def __call__(self, img: torch.Tensor, *args, **kwds):
    segmentation_map = self.yolo(img)
    
    return segmentation_map