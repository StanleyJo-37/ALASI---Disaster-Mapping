from typing import Optional

import torch

class UncertaintyLossWeighting(torch.nn.Module):
  def __init__(self):
    super(UncertaintyLossWeighting, self).__init__()
    
    self.alpha = torch.nn.Parameter(torch.ones(1))
    self.beta = torch.nn.Parameter(torch.ones(1))
    self.gamma = torch.nn.Parameter(torch.ones(1))
    
  
  def forward(
    self,
    loss_seg: torch.Tensor,
    loss_depth: Optional[torch.Tensor] = None,
    loss_normal: Optional[torch.Tensor] = None
  ):
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