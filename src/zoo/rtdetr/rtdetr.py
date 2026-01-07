"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 
from src.zoo.rtdetr.cbam import CBAM
from .adown import ADown
from .hybrid_encoder import CSPRepLayer, ConvNormLayer

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        # self.adown = ADown(256, 512)
        # self.csp1 = CSPRepLayer(in_channels=1024, out_channels=512)
        # self.csp2 = CSPRepLayer(in_channels=512, out_channels=256) #512+256
        # self.cbam = CBAM(in_channels=256) 
        # self.conv = ConvNormLayer(ch_in=512, ch_out=256, kernel_size=3, stride=1)
        # self.cbam2 = CBAM(in_channels=256)
        # self.cbam3 = CBAM(in_channels=256)
        
    def forward(self, x, targets=None):
        x = self.backbone(x) #[64, 128, 256, 512]
        # x[0] = self.adown(x[0]) # [B, 64, 160, 160] -> [B, 128, 80, 80]
        # x[1] = self.csp1(torch.cat([x[0], x[1]], dim=1)) # [B, 256, 80, 80] -> [B, 128, 80, 80]
        x = self.encoder(x)    # x[1]: [B, 128, 80, 80]  -> [B, 256, 80, 80] 
        # x[0] = self.conv(x[0]) # [256, 80, 80]
        # x[1] = self.csp2(torch.cat([x[0], x[1]], dim=1)) # [B, 384, 80, 80] -> [B, 256, 80, 80] 
        # x[1] = self.cbam(x[1])
        # x[1] = self.cbam2(x[1])
        # x[2] = self.cbam3(x[2])
        # x: [256, 80, 80], [256, 80, 80], [256, 40, 40], [256, 20, 20]
        x, enc_topk_memory = self.decoder(x, targets)

        return x, enc_topk_memory
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
