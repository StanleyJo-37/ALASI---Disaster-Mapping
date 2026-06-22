from math import inf
from typing import Literal, Dict, Any
import os

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
    mode: Literal['min', 'max'] = 'min',
    patience: int = 50,
    delta: float = 0.05,
    save_per_epoch = 10,
  ):
    self.epoch_since_last_improvement = 0
    self.best_evaluation_score = inf if mode == 'min' else -inf
    self.epoch = 0
    
    self.mode = mode
    self.patience = patience
    self.delta = delta
    self.save_per_epoch = save_per_epoch
    
    self.best_parameters = None
    self.best_loss_balancer_parameters = None
    self.saved_parameters = []
    self.saved_loss_balancer_parameters = []
    
  def record_and_check_if_halt(
    self,
    new_evaluation_score: float,
    parameters: Dict[str, Any],
    loss_balancer_parameters: Dict[str, Any]
  ):
    self.epoch += 1
    if self.mode == 'min':
      improved = new_evaluation_score < (self.best_evaluation_score - self.delta)
    else:
      improved = new_evaluation_score > (self.best_evaluation_score + self.delta)
    
    if self.epoch % self.save_per_epoch == 0:
      self.saved_parameters.append(parameters)
      self.saved_loss_balancer_parameters.append(loss_balancer_parameters)
    
    if improved:
      self.best_evaluation_score = new_evaluation_score
      self.epoch_since_last_improvement = 0
      self.best_parameters = parameters
      self.best_loss_balancer_parameters = loss_balancer_parameters
    else:
      self.epoch_since_last_improvement += 1
      
      if self.epoch_since_last_improvement > self.patience:
        return True
    
    return False
  
  def save_weights(
    self,
    save_dir:str,
    model_prefix: str,
    last_state_dict: Dict[str, Any],
    last_loss_balancer_state_dict: Dict[str, Any]
  ) -> None:
    os.makedirs('eval_results', exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'weights/model'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'weights/loss-balancer'), exist_ok=True)
    
    best_path = os.path.join(save_dir, f"weights/model/{model_prefix}_best.pth")
    torch.save(self.best_parameters, best_path)
    print(f"Saved best weights to {best_path}")
    
    best_path = os.path.join(save_dir, f"weights/loss-balancer/{model_prefix}_best.pth")
    torch.save(self.best_loss_balancer_parameters, best_path)
    print(f"Saved best loss balancer weights to {best_path}")
    
    last_path = os.path.join(save_dir, f"weights/model/{model_prefix}_last.pth")
    torch.save(last_state_dict, last_path)
    print(f"Saved last weights to {last_path}")
    
    best_path = os.path.join(save_dir, f"weights/loss-balancer/{model_prefix}_last.pth")
    torch.save(last_loss_balancer_state_dict, best_path)
    print(f"Saved last loss balancer weights to {best_path}")
    
    for i, (parameter, loss_balancer_parameters) in enumerate(zip(self.saved_parameters, self.saved_loss_balancer_parameters)):
      ckpt_path = os.path.join(save_dir, f'weights/model/{model_prefix}_ckpt_{i + 1}.pth')
      torch.save(parameter, ckpt_path)
      
      ckpt_path = os.path.join(save_dir, f'weights/loss-balancer/{model_prefix}_ckpt_{i + 1}.pth')
      torch.save(loss_balancer_parameters, ckpt_path)
    print(f"Saved weight checkpoints.")
  
  def is_checkpoint(self) -> bool:
    return not (self.epoch % self.save_per_epoch)
  
  def __delete__(self, instance):
    del self.best_parameters
    
    for (parameter, loss_balancer_parameter) in zip(self.saved_parameters, self.saved_loss_balancer_parameters):
      del parameter
      del loss_balancer_parameter
  
  def __del__(self):
    del self.best_parameters
    
    for (parameter, loss_balancer_parameter) in zip(self.saved_parameters, self.saved_loss_balancer_parameters):
      del parameter
      del loss_balancer_parameter