import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PatchMergingV2
from einops import rearrange

class Conv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class DeConv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2),
            nn.Conv2d(out_planes, out_planes, kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, image_size, patch_size, dropout):
        super().__init__()
        self.n_patches = [image_size[0]//patch_size, image_size[1]// patch_size]
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv2d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, *self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.transpose(1, -1).contiguous()
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Attention_block(nn.Module):
    '''
    Attention_block from https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
    '''
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, image_size, patch_size, mlp_hidden):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int((image_size[0] * image_size[1] ) / (patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, mlp_hidden)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        key = self.key(x)
        value = self.value(x)
        query = self.query(x)
        x, weights = self.attn(query, key, value)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, image_size, patch_size, num_heads, num_layers, dropout, extract_layers, mlp_hidden):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, image_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.merge = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        
        pre_extract_layer = 0
        for i, extract_layer in enumerate(extract_layers):
            layer_embed_dim = embed_dim*2**i
            layer_image_size = [size//2**(i+1) for size in image_size]
            for _ in range(extract_layer-pre_extract_layer):
                layer = TransformerBlock(layer_embed_dim, num_heads[i], dropout, layer_image_size, patch_size, mlp_hidden[i])
                self.layer.append(copy.deepcopy(layer))
            layer = PatchMergingV2(layer_embed_dim,spatial_dims=2)
            self.merge.append(copy.deepcopy(layer))
            pre_extract_layer = extract_layer
            
    def forward(self, x):
        hidden_states = self.embeddings(x)
        extract_layers = [hidden_states]
        merge_count = 0
        for depth, layer_block in enumerate(self.layer):
            b, h, w, c = hidden_states.shape
            hidden_states = rearrange(hidden_states, 'b h w c -> b (h w) c').contiguous()
            hidden_states, _ = layer_block(hidden_states)
            hidden_states = hidden_states.view(b, h, w, -1)
            if depth + 1 in self.extract_layers:
                hidden_states = self.merge[merge_count](hidden_states)
                extract_layers.append(hidden_states)
                merge_count += 1

        return extract_layers


class INTRANET(nn.Module):
    def __init__(self, img_shape=(224, 224), 
                 input_dim=10, 
                 output_dim=4, 
                 embed_dim=48, 
                 patch_size=2, 
                 num_heads=12, 
                 dropout=0.1,
                 mlp_hidden=[128,256,512,1024],
                 num_layers=12,
                 ext_layers=[3, 6, 9, 12]):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_dim = [x // patch_size for x in img_shape]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                num_layers,
                dropout,
                ext_layers,
                mlp_hidden
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv2dBlock(input_dim, embed_dim//2, 3),
                Conv2dBlock(embed_dim//2, embed_dim, 3)
            )

        self.decoder1 = \
            nn.Sequential(
                Conv2dBlock(embed_dim, embed_dim, 3),
                Conv2dBlock(embed_dim, embed_dim, 3)
            )

        self.decoder2 = \
            nn.Sequential(
                Conv2dBlock(embed_dim*2, embed_dim*2),
                Conv2dBlock(embed_dim*2, embed_dim*2)
            )

        self.decoder3 = \
            nn.Sequential(
                Conv2dBlock(embed_dim*4, embed_dim*4),
                Conv2dBlock(embed_dim*4, embed_dim*4)
            )

        self.decoder4 = \
            nn.Sequential(
                Conv2dBlock(embed_dim*8, embed_dim*8),
                Conv2dBlock(embed_dim*8, embed_dim*8)
            )
        
        self.decoder5 = \
            nn.Sequential(
                Conv2dBlock(embed_dim*16, embed_dim*16),
                Conv2dBlock(embed_dim*16, embed_dim*16)
            )
        
        self.decoder5_upsampler = \
            nn.ConvTranspose2d(embed_dim*16, embed_dim*8, kernel_size=2, stride=2)

        self.decoder4_upsampler = \
            nn.Sequential(
                Conv2dBlock(embed_dim*16, embed_dim*8),
                Conv2dBlock(embed_dim*8, embed_dim*8),
                nn.ConvTranspose2d(embed_dim*8, embed_dim*4, kernel_size=2, stride=2)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv2dBlock(embed_dim*8, embed_dim*4),
                Conv2dBlock(embed_dim*4, embed_dim*4),
                nn.ConvTranspose2d(embed_dim*4, embed_dim*2, kernel_size=2, stride=2)
            )

        self.decoder2_upsampler = \
            nn.Sequential(
                Conv2dBlock(embed_dim*4, embed_dim*2),
                Conv2dBlock(embed_dim*2, embed_dim*2),
                nn.ConvTranspose2d(embed_dim*2, embed_dim, kernel_size=2, stride=2)
            )

        self.decoder1_upsampler = \
            nn.Sequential(
                Conv2dBlock(embed_dim*2, embed_dim),
                Conv2dBlock(embed_dim, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv2dBlock(embed_dim*2, embed_dim),
                Conv2dBlock(embed_dim, 32),
                nn.Conv2d(32, output_dim, kernel_size=1),
            )
        self.Att5 = Attention_block(F_g=embed_dim*8,F_l=embed_dim*8,F_int=embed_dim*4)
        self.Att4 = Attention_block(F_g=embed_dim*4,F_l=embed_dim*4,F_int=embed_dim*2)
        self.Att3 = Attention_block(F_g=embed_dim*2,F_l=embed_dim*2,F_int=embed_dim)
        self.Att2 = Attention_block(F_g=embed_dim,F_l=embed_dim,F_int=embed_dim//2)
        self.Att1 = Attention_block(F_g=embed_dim,F_l=embed_dim,F_int=embed_dim//2)

    def forward(self, x):
        z = self.transformer(x)
        z0, z1, z2, z3, z4, z5 = x, *z
        z1 = rearrange(z1, 'b h w c -> b c h w').contiguous()
        z2 = rearrange(z2, 'b h w c -> b c h w').contiguous()
        z3 = rearrange(z3, 'b h w c -> b c h w').contiguous()
        z4 = rearrange(z4, 'b h w c -> b c h w').contiguous()
        z5 = rearrange(z5, 'b h w c -> b c h w').contiguous()

        z5 = self.decoder5_upsampler(z5)
        z4 = self.decoder4(z4)
        z4 = self.decoder4_upsampler(torch.cat([self.Att5(z5,z4), z5], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([self.Att4(z4,z3), z4], dim=1))
        z2 = self.decoder2(z2)
        z2 = self.decoder2_upsampler(torch.cat([self.Att3(z3,z2), z3], dim=1))
        z1 = self.decoder1(z1)
        z1 = self.decoder1_upsampler(torch.cat([self.Att2(z2,z1), z2], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([self.Att1(z1,z0), z1], dim=1))

        return output


