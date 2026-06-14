import cv2
import numpy as np
from pathlib import Path

def convert_png_to_yolo_label(npy_path: str, save_dir: str) -> list[list[float]]:
  path_obj = Path(npy_path)
  save_file = f'{save_dir}/{path_obj.stem.replace("_lab", "")}.txt'
  
  yolo_annotations = []
  label = np.load(npy_path, allow_pickle=True)
  height, width = label.shape

  unique_classes = np.unique(label)
  unique_classes = unique_classes[unique_classes != 0]
  
  for class_id in unique_classes:
    class_mask = np.where(label == class_id, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
      if cv2.contourArea(cnt) < 10:
        continue
      
      coordinates = cnt.squeeze()
      normalized_coords = [f"{(x / width):.6f} {(y / height):.6f}" for x, y in coordinates]

      poly_string = f"{int(class_id)} " + " ".join(normalized_coords)
      yolo_annotations.append(poly_string)
  
  if yolo_annotations:
    with open(save_file, 'w') as f:
      f.write('\n'.join(yolo_annotations))
          
  return yolo_annotations