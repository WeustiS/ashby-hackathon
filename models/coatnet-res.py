import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

from torchsummary import summary
import math 


def conv_3x3_bn(inp, oup, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv3d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, z, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, downsample=False, upsample=False, expansion=2):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)
        if self.upsample:
            self.proj = nn.ConvTranspose3d(inp, oup, 2, 2, 0, bias=False)
        if self.downsample:
            self.pool = nn.MaxPool3d(3, 2, 1)
            self.proj = nn.Conv3d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            if self.upsample:
                self.conv = nn.Sequential(
                        # pw
                        # down-sample in the first conv
                        nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                        nn.BatchNorm3d(hidden_dim),
                        nn.GELU(),
                        # dw
                        nn.Conv3d(hidden_dim, hidden_dim, 3, 1, 1,
                                   groups=hidden_dim, bias=False),
                        nn.BatchNorm3d(hidden_dim),
                        nn.GELU(),
                        SE(inp, hidden_dim),
                    # pw-linear
                        nn.ConvTranspose3d(hidden_dim, oup, 2, 2, 0, bias=False),
                        nn.BatchNorm3d(oup),
                )
            else:
                self.conv = nn.Sequential(
                        # pw
                        # down-sample in the first conv
                        nn.Conv3d(inp, hidden_dim, 3, stride, 1, bias=False),
                        nn.BatchNorm3d(hidden_dim),
                        nn.GELU(),
                        # dw
                        nn.Conv3d(hidden_dim, hidden_dim, 3, 1, 1,
                                   groups=hidden_dim, bias=False),
                        nn.BatchNorm3d(hidden_dim),
                        nn.GELU(),
                        SE(inp, hidden_dim),
                    # pw-linear
                        nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                        nn.BatchNorm3d(oup),
                )
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm3d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        elif self.upsample:
            return self.proj(x) + self.conv(x)
        else:
            return x + self.conv(x)



class CoAtNet_3d(nn.Module):
    def __init__(self, img_size, dims, depths,  num_classes=10):
        super().__init__()
        self.img_size = img_size
        c, z, h, w = img_size
                                                                # 48 160 176
#         # 16
#         self.s0 = self._make_layer(c,       dims[0], depths[0]) # 24, 80 80 
#         self.s1 = self._make_layer(dims[0], dims[1], depths[1]) # 12 40 40  
#         self.s2 = self._make_layer(dims[1], dims[2], depths[2]) # 6 20 20  
#         self.s3 = self._make_layer(dims[2], dims[3], depths[3]) # 3 10 11  -> 384
        
        self.encoder = nn.Sequential(
            nn.Conv3d(c, dims[0], (1,7,7), (1,2,2), padding=(0,3,3)),
            self._make_layer(dims[0], dims[1], depths[0]),
            self._make_layer(dims[1], dims[2], depths[1]),
            self._make_layer(dims[2], dims[3], depths[2]),
            self._make_layer(dims[3], dims[4], depths[3])
        )
        
        self.decoder = nn.Sequential(
            self._make_layer(dims[4], dims[5], depths[4], downsample=False),
            self._make_layer(dims[5], dims[6], depths[5], downsample=False),
            self._make_layer(dims[6], dims[7], depths[6], downsample=False),
            self._make_layer(dims[7], dims[8], depths[7], downsample=False),
            self._make_layer(dims[8], num_classes, depths[8], downsample=False),
            
        )
        
        self.bottleneck_shape = (dims[4], z//16, h//32, math.ceil(w/32))
        
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def _make_layer(self, inp, oup, depth, downsample=True):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                #down/up sample
                if downsample:
                    layers.append(MBConv(inp, oup, downsample=True))
                else:
                    layers.append(MBConv(inp, oup, upsample=True))
            else:
                layers.append(MBConv(oup, oup))
        return nn.Sequential(*layers)

if __name__ == "__main__":
    print("Making Model")
    model = CoAtNet_3d((101, 48, 160, 160), [16]*9, [2]*8).cuda()
    summary(model, (101, 48, 160, 176))
    