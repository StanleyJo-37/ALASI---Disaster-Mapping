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

def precision(pred: np.array, truth: np.array):
  tp = np.sum(pred & truth)
  fp = np.sum(pred & ~truth)
  
  if tp + fp == 0:
    return 1.0 if np.sum(truth) == 0 else 0.0
  
  return tp / (tp + fp)

def recall(pred: np.array, truth: np.array):
  tp = np.sum(pred & truth)
  fn = np.sum(~pred & truth)
  
  if tp + fn == 0:
    return 1.0 if np.sum(pred) == 0 else 0.0
  
  return tp / (tp + fn)

def f1_score(pred: np.array, truth: np.array):
  p = precision(pred, truth)
  r = recall(pred, truth)
  
  if p + r == 0:
    return 0.0
  
  return 2 * p * r / (p + r)

def dice_coefficient(pred: np.array, truth: np.array):  
  intersect = np.sum(pred & truth)
  union = np.sum(pred) + np.sum(truth)

  if union == 0:
    return 1.0 if intersect == 0 else 0.0
  
  return 2 * intersect / union

def hausdorff_distance(pred: np.array, truth: np.array, resolution_per_pixel: float):
  # dihitung per gambar (H, W), lalu dirata-rata di batch
  distances = []
  
  for pred_img, truth_img in zip(pred, truth):
    if np.sum(pred_img) == 0 and np.sum(truth_img) == 0:
      distances.append(0.0)
      continue
    
    if np.sum(pred_img) == 0 or np.sum(truth_img) == 0:
      image_diag = np.sqrt(pred_img.shape[0]**2 + pred_img.shape[1]**2)
      distances.append(image_diag * resolution_per_pixel)
      continue
    
    dist_pixel = skimage_hd(pred_img, truth_img)
    distances.append(dist_pixel * resolution_per_pixel)
  
  return np.mean(distances)

def evaluate(pred_mask: np.array, true_mask: np.array, labels: list, resolution_per_pixel: float):
  results = {}
  
  for label in labels:
    pred_mask_binary = np.array(pred_mask == label).astype(bool)
    true_mask_binary = np.array(true_mask == label).astype(bool)
    
    acc = accuracy(pred_mask_binary, true_mask_binary)
    iou = intersect_over_union(pred_mask_binary, true_mask_binary)
    prec = precision(pred_mask_binary, true_mask_binary)
    rec = recall(pred_mask_binary, true_mask_binary)
    f1 = f1_score(pred_mask_binary, true_mask_binary)
    dice = dice_coefficient(pred_mask_binary, true_mask_binary)
    hd = hausdorff_distance(pred_mask_binary, true_mask_binary, resolution_per_pixel)
    
    results[label] = {
      'Accuracy': acc,
      'Intersect Over Union': iou,
      'Precision': prec,
      'Recall': rec,
      'F1': f1,
      'Dice Coefficient': dice,
      'Hausdorff Distance': hd,
    }
  
  # rata-rata antar kelas (macro); 'Intersect Over Union' di sini = mIoU
  results['Mean'] = {
    metric: np.mean([results[label][metric] for label in labels])
    for metric in results[labels[0]]
  }
  
  # pixel accuracy global: semua kelas sekaligus
  results['Pixel Accuracy'] = accuracy(pred_mask, true_mask)
  
  return results
