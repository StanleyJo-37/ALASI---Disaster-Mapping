from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

class DecoderHead(nn.Module):
  def __init__(self, out_channels, target_size = (640, 640), *args, **kwargs):
    super(DecoderHead, self).__init__(*args, **kwargs)
    
    self.target_size = target_size
    
    self.decoder_block_1 = self.create_decoder_block(512, 256, 384) # in: 512, 20, 20 -> out: 256, 40, 40
    self.decoder_block_2 = self.create_decoder_block(768, 256, 512) # in: 256, 40, 40 concat (512, 40, 40) -> 768, 40, 40 -> out: 256, 80, 80
    self.decoder_block_3 = self.create_decoder_block(768, 256, 512) # in: 256, 80, 80 concat (512, 80, 80) -> 768, 80, 80 -> out: 256, 160, 160
    self.decoder_block_4 = self.create_decoder_block(256, 128) # in: 256, 160, 160 -> out: 128, 320, 320
    self.output_block = nn.Sequential(
      self.create_output_block(128, 64, 3, 1), # 64, 320, 320
      self.create_output_block(64, 32, 3, 1), # 32, 320, 320
      self.create_output_block(32, 16, 3, 1), # 16, 320, 320
      
      nn.Conv2d(16, out_channels, kernel_size=1) # out, 320, 320
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
    
  def _icnr_init(
    self,
    tensor: torch.Tensor,
    upscale_factor: int = 2,
  ):
    r = upscale_factor ** 2
    sub_kernel = torch.zeros((tensor.shape[0] // r, *tensor.shape[1:]))
    torch.nn.init.kaiming_normal_(sub_kernel)
    kernel = sub_kernel.repeat_interleave(r, dim=0)
    with torch.no_grad():
      tensor.copy_(kernel)
    
  def create_decoder_block(
    self,
    in_channels: int,
    out_channels: int,
    mid_channels: Optional[int] = None,
  ) -> nn.Module:
    if not mid_channels:
      mid_channels = out_channels
    
    PixelShuffleConv2d = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=3, padding=1, bias=False)
    self._icnr_init(PixelShuffleConv2d.weight)
    
    return nn.Sequential(
      PixelShuffleConv2d,
      nn.PixelShuffle(2),
      nn.SiLU(inplace=True),
      
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
    x = self.decoder_block_4(x)   # upsample
    
    out = self.output_block(x)
    out_interpolated = F.interpolate(out, self.target_size, mode='bilinear')
    
    return out_interpolated