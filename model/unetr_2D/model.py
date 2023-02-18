import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        self.n_patches = int((image_size[0] * image_size[1] ) / (patch_size  * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv2d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


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
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, image_size, patch_size, mlp_hidden)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class UNETR(nn.Module):
    def __init__(self, img_shape=(128, 128), 
                 input_dim=19, 
                 output_dim=11, 
                 embed_dim=768, 
                 patch_size=16, 
                 num_heads=12, 
                 dropout=0.1,
                 mlp_hidden=2048,
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
                Conv2dBlock(input_dim, 32, 3),
                Conv2dBlock(32, 64, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                DeConv2dBlock(embed_dim, 512),
                DeConv2dBlock(512, 256),
                DeConv2dBlock(256, 128)
            )

        self.decoder6 = \
            nn.Sequential(
                DeConv2dBlock(embed_dim, 512),
                DeConv2dBlock(512, 256),
            )

        self.decoder9 = \
            DeConv2dBlock(embed_dim, 512)

        self.decoder12_upsampler = \
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv2dBlock(1024, 512),
                Conv2dBlock(512, 512),
                Conv2dBlock(512, 512),
                nn.ConvTranspose2d(512, 516, kernel_size=2, stride=2)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv2dBlock(512, 256),
                Conv2dBlock(256, 256),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv2dBlock(256, 128),
                Conv2dBlock(128, 128),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv2dBlock(128, 64),
                Conv2dBlock(64, 64),
                nn.Conv2d(64, output_dim, kernel_size=1),
            )

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output