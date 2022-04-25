import torch
import torch.nn as nn
from models.coatnet import CoAtNet_3d
from torchsummary import summary
import math
from einops import rearrange
from einops.layers.torch import Rearrange

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, img_size, dims, blocks, tok_dim, heads, encoder_layers, bn_model, num_classes=10):
        super().__init__()
        c, t, z, h, w = img_size
        
        # load BottleNeckPredictor
        #                           img_shape       dims    blocks
        bn_shape = bn_model.bottleneck_shape # (dims[4], 3, 5, 6) || (B, dims[4]*90)
        bn_dim = bn_shape[0] # *bn_shape[1]*bn_shape[2]*bn_shape[3]
        bn_img_dim = bn_shape[0]*bn_shape[1]*bn_shape[2]*bn_shape[3]
        
        print(bn_shape, bn_dim)
        self.tok_dim = tok_dim
        self.img_size = img_size
        self.bn_shape = bn_shape
        self.bn_dim = bn_dim
        self.num_classes= num_classes
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.bottleneck_to_tok = nn.Linear(bn_dim, tok_dim)
        print("BN2Tok:", "{:,}".format(count_params(self.bottleneck_to_tok)))
        self.pos_emb = PositionalEncoding(tok_dim, max_len=t)
        
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(tok_dim, heads, tok_dim*2, activation='gelu', batch_first=True), encoder_layers)
        print("Transformer:", "{:,}".format(count_params(self.transformer)))
        self.tok_to_bottleneck = nn.Sequential(
           # nn.ConvTranspose3d(tok_dim, bn_shape[0], (bn_shape[1], bn_shape[2], bn_shape[3]), (1, 1, 1), (0, 0, 0), bias=False)
            nn.Linear(tok_dim, bn_img_dim),
        )
#         self.tok_to_bottleneck = nn.Sequential(
#             nn.Linear(tok_dim, tok_dim*4),
#             Rearrange("b t (c z h w) -> (b t) c z h w",
#                      c=tok_dim//2,
#                      z=2,
#                      h=2,
#                      w=2),
#             torch.nn.ConvTranspose3d(tok_dim//2, bn_dim, (2, 3, 2), (1, 2, 4), (0, 0, 0), bias=False)
#         ) 
        
        print("Tok2BN:", "{:,}".format(count_params(self.tok_to_bottleneck)))
        self.img_decoder = bn_model.decoder
        print("Decoder:", "{:,}".format(count_params(self.img_decoder)))
        
    def forward(self, x):
        b, t = x.shape[:2]
        x = x.view(b*t, 64, 3, 5, 6)
        tmp = x 
        x = self.avg_pool(x)  # bt, c, 1, 1, 1
        x = x.view(b, t, self.bn_dim) # b, t, c
        
        
        b = len(x)
        c, t, z, h, w = self.img_size
        
        
        # B, T, E
        x = self.bottleneck_to_tok(x)
       # print("Transformer")
        x = self.pos_emb(x)
        
        
        x = self.transformer(x)
        x = x[:, :self.num_classes]
        # B nc e
       #  x = x.view(b*self.num_classes, self.tok_dim, 1, 1, 1)
        # B, num_classes, E -> 
       # print("To Bn")
        x = self.tok_to_bottleneck(x)
        x = x.view(b*t, 64, 3, 5, 6)
       
        #print("Decoding")
        # B, num_classes, E
        # B C Z H W 
#         x = rearrange(x, "b t (c2 z h w) -> (b t) c2 z h w",
#                       c2=self.bn_shape[0],
#                       z=self.bn_shape[1],
#                       h=self.bn_shape[2],
#                       w=self.bn_shape[3])
        
        x += tmp
        x = x.cuda(3)
        x = self.img_decoder(x)

        x = rearrange(x, "(b t) c z h w -> b (t c) z h w",
                      b=b,
                      t=self.num_classes
                      # c = 1
                     )
        
        return x
            
if __name__ == "__main__":
    c = 102 # 102
    t = 15 # 145
    emb_dim = 256
    channels = [16, 32, 64, 128, emb_dim, 128, 32, 16, 12]
    blocks = [4, 4, 4, 4, 3, 2, 1, 1]
    model = Model((c, t, 48, 160, 176), 
                  channels,
                 [2, 2, 3, 4, 4, 3, 2, 1],
                 tok_dim=256,
                 heads=8,
                 encoder_layers=8).cuda()
    e = model.bn_dim 
    print("----------Created-------------")
    x = torch.rand(2, t, emb_dim).cuda()
    y = model(x)
    print(y.shape)
    summary(model, (t, emb_dim))
    # img_size, dims, blocks, tok_dim, heads, encoder_layers, num_classes=10