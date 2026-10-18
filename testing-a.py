import re

import numpy as np
from PIL import Image, ImageFile
import xml.etree.ElementTree as ET

focal_length_regex = r'drone-dji:CalibratedFocalLength="([\d.]+)"'
center_x_regex = r'drone-dji:CalibratedOpticalCenterX="([\d.]+)"'
center_y_regex = r'drone-dji:CalibratedOpticalCenterY="([\d.]+)"'

def get_camera_intrinsics(img: ImageFile.ImageFile):
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
    return {
      "fx": DEFAULT_FX,
      "fy": DEFAULT_FY,
      "cx": DEFAULT_CX,
      "cy": DEFAULT_CY,
      "width": width,
      "height": height
    }

  xmp_str = xmp_data.decode('utf-8', errors='ignore') if isinstance(xmp_data, bytes) else xmp_data
  
  focal_length = float(re.findall(focal_length_regex, xmp_str)[0])
  center_x = float(re.findall(center_x_regex, xmp_str)[0])
  center_y = float(re.findall(center_y_regex, xmp_str)[0])
  
  if np.isnan(focal_length) or np.isnan(center_x) or np.isnan(center_y):
    return {
      "fx": np.where(np.isnan(focal_length), DEFAULT_FX, focal_length),
      "fy": np.where(np.isnan(focal_length), DEFAULT_FY, focal_length),
      "cx": np.where(np.isnan(center_x), DEFAULT_CX, center_x),
      "cy": np.where(np.isnan(center_y), DEFAULT_CY, center_y),
      "width": width,
      "height": height
    }
  
  return {
    "fx": focal_length,
    "fy": focal_length,
    "cx": center_x,
    "cy": center_y,
    "width": width,
    "height": height
  }

with Image.open('./ai/data/RescueNet/train/train-org-img/10786.jpg') as img:
  print(get_camera_intrinsics(img))