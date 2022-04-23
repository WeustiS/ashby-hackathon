import torch
import torch.nn as nn
from convNd import convNd
from coatnet import CoAtNet

class Model(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        c, t, z, h, w = img_size
        # (self, image_size, in_channels, num_blocks, channels, num_classes=1000, block_types=['C', 'C', 'T', 'T']):
        self.img_encoder = CoAtNet((h,w), c, [2,2,2,2,1], [64, 96, 128, 192, 256])
    
    def forward(self,x):
        pass

if __name__ == "__main__":
    # B, C, T, Z, H, W
    # C in range [1, 101]
    # T in range 145
    # z in range 39 (z is pressure, NOT altitude)
    # H in range 159
    # W in range 169
    x = torch.rand(1, 101, 145, 39, 159, 169)