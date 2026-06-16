from typing import Optional

import torch

class UncertaintyLossWeighting(torch.nn.Module):
  def __init__(self, device: Optional[str] = None):
    super(UncertaintyLossWeighting, self).__init__()
    
    self.alpha = torch.nn.Parameter(torch.ones(1, device=device))
    self.beta = torch.nn.Parameter(torch.ones(1, device=device))
    self.gamma = torch.nn.Parameter(torch.ones(1, device=device))
  
  def forward(self, loss_seg, loss_depth, loss_normal):
    precision_seg = torch.exp(-self.alpha)
    l_seg = precision_seg * loss_seg + self.alpha
    
    precision_depth = torch.exp(-self.beta)
    l_depth = precision_depth * loss_depth + self.beta
    
    precision_normal = torch.exp(-self.gamma)
    l_normal = precision_normal * loss_normal + self.gamma
    
    return l_seg + l_depth + l_normal