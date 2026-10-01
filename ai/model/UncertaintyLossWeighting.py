from typing import Optional

import torch

class UncertaintyLossWeighting(torch.nn.Module):
  """Homoscedastic uncertainty weighting (Kendall, Gal & Cipolla, 2018).

  Each task learns s = log(sigma^2):
    classification (seg):        exp(-s) * L + 0.5 * s
    regression (depth, normal):  0.5 * exp(-s) * L + 0.5 * s
  """

  def __init__(self, init_log_var: float = 1.0):
    super(UncertaintyLossWeighting, self).__init__()
    
    self.alpha = torch.nn.Parameter(torch.full((1,), init_log_var))
    self.beta = torch.nn.Parameter(torch.full((1,), init_log_var))
    self.gamma = torch.nn.Parameter(torch.full((1,), init_log_var))
  
  def forward(
    self,
    loss_seg: torch.Tensor,
    loss_depth: Optional[torch.Tensor] = None,
    loss_normal: Optional[torch.Tensor] = None
  ):
    zero = loss_seg.new_zeros(())
    weighted_depth_loss = zero
    weighted_normal_loss = zero

    # Weighted Semantic Segmentation Loss
    weighted_seg_loss = torch.exp(-self.alpha) * loss_seg + (self.alpha * 0.5)
    loss_total = weighted_seg_loss
    
    # Weighted Depth Loss
    if loss_depth is not None:
      weighted_depth_loss = 0.5 * ((torch.exp(-self.beta) * loss_depth) + self.beta)
      loss_total = loss_total + weighted_depth_loss
    
    # Weighted Normal Loss
    if loss_normal is not None:
      weighted_normal_loss = 0.5 * ((torch.exp(-self.gamma) * loss_normal) + self.gamma)
      loss_total = loss_total + weighted_normal_loss
    
    # Returns the Sum of Weighted Loss
    return loss_total, weighted_seg_loss, weighted_depth_loss, weighted_normal_loss