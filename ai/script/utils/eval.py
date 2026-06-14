import numpy as np
from skimage.metrics import hausdorff_distance as skimage_hd

def accuracy(pred: np.array, truth: np.array):
  true = np.sum(pred == truth)
  false = np.sum(pred != truth)
  
  accuracy = true / (true + false)
  
  return accuracy

def intersect_over_union(pred: np.array, truth: np.array):
  intersect = np.sum(pred & truth)
  union = np.sum(pred | truth)
  
  if union == 0:
    return 1.0 if intersect == 0 else 0.0
  
  return intersect / union

def dice_coefficient(pred: np.array, truth: np.array):  
  intersect = np.sum(pred & truth)
  union = np.sum(pred) + np.sum(truth)

  if union == 0:
    return 1.0 if intersect == 0 else 0.0
  
  return 2 * intersect / union

def hausdorff_distance(pred: np.array, truth: np.array, resolution_per_pixel: float):
  if np.sum(pred) == 0 or np.sum(truth) == 0:
    image_diag = np.sqrt(pred.shape[0]**2 + pred.shape[1]**2)
    return image_diag * resolution_per_pixel
  
  dist_pixel = skimage_hd(pred, truth)
  dist_metres = dist_pixel * resolution_per_pixel
  
  return dist_metres

def evaluate(pred_mask: np.array, true_mask: np.array, labels: list, resolution_per_pixel: float):
  results = {}
  
  for label in labels:
    pred_mask_binary = np.array(pred_mask == label).astype(bool)
    true_mask_binary = np.array(true_mask == label).astype(bool)
    
    acc = accuracy(pred_mask_binary, true_mask_binary)
    iou = intersect_over_union(pred_mask_binary, true_mask_binary)
    dice = dice_coefficient(pred_mask_binary, true_mask_binary)
    hd = hausdorff_distance(pred_mask_binary, true_mask_binary, resolution_per_pixel)
    
    results[label] = {
      'Accuracy': acc,
      'Intersect Over Union': iou,
      'Dice Coefficient': dice,
      'Hausdorff Distance': hd,
    }
  
  return results
