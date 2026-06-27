from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

class DecoderHead(nn.Module):
  def __init__(self, out_channels, target_size = (640, 640), *args, **kwargs):
    super(DecoderHead, self).__init__(*args, **kwargs)
    
    self.target_size = target_size
    
    self.decoder_block_1 = self.create_decoder_block(512, 256, 384) # in: 512, 20, 20 -> out: 256, 40, 40
    self.decoder_block_2 = self.create_decoder_block(768, 256, 512) # in: 256, 40, 40 concat (512, 40, 40) -> 768, 40, 40 -> 256, 80, 80
    self.decoder_block_3 = self.create_decoder_block(768, 256, 512) # in: 256, 80, 80 concat (512, 80, 90) -> 768, 80, 80 -> 256, 160, 160
    self.output_block = nn.Sequential(
      self.create_output_block(256, 128, 3, 1), # 128, 160, 160
      self.create_output_block(128, 64, 3, 1), # 64, 160, 160
      self.create_output_block(64, 32, 3, 1), # 32, 160, 160
      self.create_output_block(32, 16, 3, 1), # 16, 160, 160
      nn.Conv2d(16, out_channels, kernel_size=1) # out, 160, 160
    )
    
  def create_output_block(
    self,
    in_channel: int,
    out_channel: int,
    kernel_size: int | tuple[int, int],
    padding: int | tuple[int, int]
  ):
    return nn.Sequential(
      nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
      nn.BatchNorm2d(out_channel),
      nn.SiLU(inplace=True),
    )
    
  def create_decoder_block(
    self,
    in_channels: int,
    out_channels: int,
    mid_channels: Optional[int],
  ) -> nn.Module:
    if not mid_channels:
      mid_channels = out_channels
    
    return nn.Sequential(
      nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
      nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(mid_channels),
      nn.SiLU(inplace=True),
      nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.SiLU(inplace=True),
    )
  
  def forward(self, p5, p4, p3):
    x = self.decoder_block_1(p5)  # upsample p5
    x = torch.cat([x, p4], dim=1) # concat skip from p4
    x = self.decoder_block_2(x)   # upsample
    x = torch.cat([x, p3], dim=1) # concat skip from p3
    x = self.decoder_block_3(x)   # upsample
    
    out = self.output_block(x)
    out_interpolated = F.interpolate(out, self.target_size, mode='bilinear')
    
    return out_interpolated