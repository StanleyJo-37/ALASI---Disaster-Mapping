from typing_extensions import Optional
from pydantic import BaseModel, ConfigDict
import numpy as np

class DisasterSiteMap(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  # The original RGB image (Shape: [H, W, 3])
  map_2d: np.ndarray = np.array([])
  
  # The YOLO Instance Segmentation Mask (Shape: [H, W])
  segmentation_map: np.ndarray = np.array([])
  
  # The Hallucinated Depth Map (Shape: [H, W])
  depth_map: Optional[np.ndarray] = None

  # 3D Transformation matrix (Shape: [4, 4])
  transformation_matrix: Optional[np.ndarray] = None
  
  # The 3D Point Cloud (Shape: [N, 6])
  # [X, Y, Z, R, G, B] or [X, Y, Z, Class_ID, Confidence, 0]
  point_cloud_3d: Optional[np.ndarray] = None