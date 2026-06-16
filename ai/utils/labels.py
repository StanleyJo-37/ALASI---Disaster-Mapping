from typing import List, Tuple

import torch
import numpy as np
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