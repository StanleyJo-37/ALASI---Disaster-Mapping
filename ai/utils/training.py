from math import inf

import torch

def compute_normal_loss(pred_norm: torch.Tensor, gt_norm: torch.Tensor):
  """ compute per-pixel surface normal error in degrees
    NOTE: pred_norm and gt_norm should be torch tensors of shape (B, 3, ...)
  """
  pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
  pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
  pred_error = torch.acos(pred_error) * 180.0 / torch.pi
  return pred_error.mean()

class EarlyStoppingAndCheckpointing():
  def __init__(
    self,
    patience: int = 50,
    delta: float = 0.05,
    save_per_epoch = 10,
  ):
    self.epoch_since_last_improvement = 0
    self.best_evaluation_score = -inf
    self.epoch = 0
    
    self.patience = patience
    self.delta = delta
    self.save_per_epoch = save_per_epoch
    
    self.best_parameters = None
    self.saved_parameters = []
    
  def record_and_check_if_halt(
    self,
    new_evaluation_score: float,
    parameters: dict[str, any]
  ):
    self.epoch += 1
    improved = new_evaluation_score > (self.best_evaluation_score + self.delta)
    
    if self.epoch % self.save_per_epoch == 0:
      self.saved_parameters.append(parameters)
    
    if improved:
      self.best_evaluation_score = new_evaluation_score
      self.epoch_since_last_improvement = 0
      self.best_parameters = parameters
    else:
      self.epoch_since_last_improvement += 1
      
      if self.epoch_since_last_improvement > self.patience:
        return True
    
    return False
  
  def save_weights(self, save_path: str) -> None:
    return save_path
  
  def is_checkpoint(self) -> bool:
    return not (self.epoch % self.save_per_epoch)
  
  def __delete__(self, instance):
    del self.best_parameters
    
    for parameter in self.saved_parameters:
      del parameter