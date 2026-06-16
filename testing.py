from typing import List

import numpy as np
import cv2

def _convert_png_to_yolo_label(label: np.array) -> List[List[float]]:
  yolo_annotations = {
    'class_ids': [],
    'bboxes': [],
    'masks': []
  }
  height, width = label.shape

  unique_classes = np.unique(label)
  unique_classes = unique_classes[unique_classes != 0]
  
  class_ids, flattened_contours, bboxes = [], [], []
  for class_id in unique_classes:
    class_mask = np.where(label == class_id, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
      if cv2.contourArea(cnt) < 10:
        continue
      
      coords = cnt.squeeze()
      if coords.ndim == 1:
        continue
      
      yolo_annotations['class_ids'].append(class_id)
      yolo_annotations['masks'].append(coords)
      
      x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
      if i == 0:
        print(x_box, y_box, w_box, h_box)
      
      x = (x_box + (w_box / 2)) / width
      y = (y_box + (h_box / 2)) / height
      w = w_box / width
      h = h_box / height
      
      yolo_annotations['bboxes'].append([x, y, w, h])
  
  return yolo_annotations
  
img = np.load('./ai/data/RescueNet/train/train-labelnpy-img/10778_lab.npy')
print(_convert_png_to_yolo_label(img)['masks'])
