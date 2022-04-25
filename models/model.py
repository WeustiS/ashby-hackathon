import torch
import torch.nn as nn
from models.convNd import convNd
# from torchsummary import summary

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(1, dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        #self.avg_pool = # nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _, _ = x.size()
        y = torch.mean(x, dim=(2,3,4,5))
        y = self.fc(y).view(b, c, 1, 1, 1, 1)
        return x * y


class MBConv(nn.Module):
    def __init__(self, inp, oup, downsample=False, upsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample
        stride = 1 if self.downsample == False and self.upsample==False else 2
        up_kernel = 1 if self.upsample == False else 2
        hidden_dim = int(inp * expansion)

        self.pool = nn.MaxPool3d(3, 2, 1)
        self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.conv = nn.Sequential(
            convNd(
                in_channels=inp,
                out_channels=hidden_dim,
                num_dims=4,
                kernel_size=up_kernel,
                stride=(stride,stride,stride,stride), # t z h w 
                padding=0,
                use_bias=False,
                is_transposed=upsample
            ),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            # dw
                
            convNd(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_dims=4,
                kernel_size=3,
                stride=(1,1,1,1), # t z h w 
                padding=1,
                use_bias=False
            ),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            SE(inp, hidden_dim),
            # pw-linear
            convNd(
                in_channels=hidden_dim,
                out_channels=oup,
                num_dims=4,
                kernel_size=1,
                stride=(1,1,1,1), # t z h w 
                padding=0,
                use_bias=False
            ),
            nn.GroupNorm(1, oup),
        )
        
        self.conv = PreNorm(inp, self.conv, nn.GroupNorm)
    def forward(self, x):
        return self.conv(x) 
        
class Model(nn.Module):
    def __init__(self, img_size, dims, depths, output_size, num_classes=10):
        super().__init__()
        self.img_size = img_size
        c, t, z, h, w = img_size
        t_out, z_out, h_out, w_out = output_size
        
        t_offset = (t-t_out)//2
        z_offset = (z-z_out)//2
        h_offset = (h-h_out)//2
        w_offset = (w-w_out)//2
        
        self.offsets = (t_offset, z_offset, h_offset, w_offset)
        self.output_size = output_size
        
        self.s0 = self._make_layer(c,       dims[0], depths[0])
        self.s1 = self._make_layer(dims[0], dims[1], depths[1])
        self.s2 = self._make_layer(dims[1], dims[2], depths[2])
        self.s3 = self._make_layer(dims[2], dims[3], depths[3])
        
        self.s3u = self._make_layer(dims[3], dims[2], depths[3], downsample=False)
        self.s2u = self._make_layer(dims[2], dims[1], depths[2], downsample=False)
        self.s1u = self._make_layer(dims[1], dims[0], depths[1], downsample=False)
        self.s0u = self._make_layer(dims[0], num_classes, depths[0], downsample=False)
        
        
    def forward(self,x):
        c, t, z, h, w = self.img_size

        s0_o = self.s0(x)
        s1_o = self.s1(s0_o)
        s2_o = self.s2(s1_o)
      #  s3_o = self.s3(s2_o)

      #  x = self.s3u(s3_o)  + s2_o
        x = self.s2u(s2_o)     + s1_o
        x = self.s1u(x)     + s0_o
        x = self.s0u(x)
        
        # crop output
        # b c t z h w 
        t_offset, z_offset, h_offset, w_offset = self.offsets 
        t_out, z_out, h_out, w_out = self.output_size
        x = x[:, :, 
              t_offset:t_out+t_offset,
              z_offset:z_out+z_offset,
              h_offset:h_out+h_offset,
              w_offset:w_out+w_offset
             ]
        
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
    # B, C, T, Z, H, W
    # C in range [1, 101]
    # T in range 145
    # z in range 39 (z is pressure, NOT altitude)
    # H in range 159
    # W in range 169
    #x = torch.rand(1, 101, 145, 39, 159, 169).cuda()
    
    model = Model((101, 145, 39, 159, 169), [32, 32, 32, 32], [4, 3, 2, 1])

   # summary(model, (101, 145, 39, 159, 169), device="cpu")
    