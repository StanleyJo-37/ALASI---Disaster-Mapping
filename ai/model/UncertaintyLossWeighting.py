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
    loss_total = torch.exp(-self.alpha) * loss_seg + self.alpha
    
    if loss_depth is not None:
      loss_total = loss_total + (torch.exp(-self.beta) * loss_depth) + self.beta
    
    if loss_normal is not None:
      loss_total = loss_total + (torch.exp(-self.gamma) * loss_normal) + self.gamma
    
    return loss_total