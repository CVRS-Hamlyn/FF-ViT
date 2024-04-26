from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.fft as fft

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

class Self_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., locality=True):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.locality = locality
        if locality:
            self.scale = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))
        else:
            self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        if self.locality:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale.exp()
            mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(mask, mask_value)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., locality=True):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.locality = locality
        if locality:
            self.scale = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))
        else:
            self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, BM_map):
        qkv = (self.to_q(x), *self.to_kv(BM_map).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if self.locality:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale.exp()
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., self_locality=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Self_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, locality=self_locality)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Fourier_Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, h, w, groups=1):
        super(Fourier_Embedding, self).__init__()
        self.groups = groups
        self.flatten = Rearrange('b c h w -> b (c h w)')
        self.linear = nn.Linear(2 * in_channels * h * (w//2 + 1), out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        bs, c, h ,w = x.size()
        ffted = fft.rfftn(x, s=(h,w), dim=(-2, -1), norm='ortho')
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)
        
        ffted = self.flatten(ffted)
        latent = self.linear(ffted)

        return self.relu(latent)

class FFC_patch_embeding(nn.Module):
    def __init__(self, f_height, f_width, patch_height, patch_width, dim=768, shift=True):
        super().__init__()
        self.shift = shift
        if shift:
            self.FFC = Fourier_Embedding(in_channels=5, out_channels=dim, h=patch_height, w=patch_width)
        else:
            self.FFC = Fourier_Embedding(in_channels=1, out_channels=dim, h=patch_height, w=patch_width)
        self.to_patch = Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_height, p2 = patch_width)
        self.to_embedded_patch = Rearrange('(b h w) c -> b (h w) c', h = f_height, w = f_width)
        self.LN = nn.LayerNorm(dim)
    def forward(self, x):
        if self.shift:
            shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
            shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
            x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
            x = self.to_patch(x_with_shifts)
        else:
            x = self.to_patch(x)
        x = self.FFC(x)
        x = self.to_embedded_patch(x)
        x = self.LN(x)

        return x

    

class Pyramid_Angle_Infer(nn.Module):
    def __init__(self, H, W, dim, self_locality, cross_locality):
        super().__init__()

        self.to_bcw0 = Rearrange('b (h w) c -> b (c h) w', h=H, w=W)
        self.to_bnc1 = Rearrange('b (c h) w -> b (h w) c', h=H//2, w=W-2)
        self.to_bcw1 = Rearrange('b (h w) c -> b (c h) w', h=H//2, w=W-2)
        self.to_bnc2 = Rearrange('b (c h) w -> b (h w) c', h=H//4, w=W-4)
        self.to_bcw2 = Rearrange('b (h w) c -> b (c h) w', h=H//4, w=W-4)
        # self.to_bnc3 = Rearrange('b (c h) w -> b (h w) c', h=H//8, w=W-6)
        # self.to_bcw3 = Rearrange('b (h w) c -> b (c h) w', h=H//8, w=W-6)
        
        self.pyramid_0 = nn.Sequential(
                nn.AvgPool1d(W//2, W//2),
                Rearrange('b c w -> b (c w)'),
                nn.LayerNorm(2*H*dim),
                nn.Linear(2*H*dim, 1),
            )
        # self.conv1_latent = nn.Conv1d(H*dim, H*dim//2, 3)
        self.latent_compression_1 = nn.Sequential(
                nn.AvgPool1d(3, 1),
                nn.Conv1d(H*dim, H*dim//2, 1)
            )
        # self.conv1_BM = nn.Conv1d(H*dim, H*dim//2, 3)
        self.BM_compression_1 = nn.Sequential(
                nn.AvgPool1d(3, 1),
                nn.Conv1d(H*dim, H*dim//2, 1)
            )
        self.pyramid_1 = nn.Sequential(
                nn.AvgPool1d((W-2)//2, (W-2)//2),
                Rearrange('b c w -> b (c w)'),
                nn.LayerNorm(H*dim),
                nn.Linear(H*dim, 1),
            )
        # self.conv2_latent = nn.Conv1d(H*dim//2, H*dim//4, 3)
        self.latent_compression_2 = nn.Sequential(
                nn.AvgPool1d(3, 1),
                nn.Conv1d(H*dim//2, H*dim//4, 1)
            )
        # self.conv2_BM = nn.Conv1d(H*dim//2, H*dim//4, 3)
        self.BM_compression_2 = nn.Sequential(
                nn.AvgPool1d(3, 1),
                nn.Conv1d(H*dim//2, H*dim//4, 1)
            )
        self.pyramid_2 = nn.Sequential(
                nn.AvgPool1d((W-4)//2, (W-4)//2),
                Rearrange('b c w -> b (c w)'),
                nn.LayerNorm(H*dim//2),
                nn.Linear(H*dim//2, 1),
            )
        

        self.cross_attn_0 = Cross_Attention(dim=dim, locality=cross_locality)
        self.cross_attn_1 = Cross_Attention(dim=dim, locality=cross_locality)
        self.self_attn_1 = Self_Attention(dim=dim, locality=self_locality)
        self.cross_attn_2 = Cross_Attention(dim=dim, locality=cross_locality)
        self.self_attn_2 = Self_Attention(dim=dim, locality=self_locality)


        self.cls_infer0 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
        self.cls_infer1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
        self.cls_infer2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
        self.cls_infer3 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self, x, BM_map):
        outputs = {}
        # print("input:", x.shape)
        # print("BM_input:", BM_map.shape)
        if BM_map != None:
            x = x + self.cross_attn_0(x, BM_map)
            BM_map = self.to_bcw0(BM_map)
        x_latent = self.to_bcw0(x[:, 1:])
        # print(x_latent.shape)
        x_cls = x[:, 0].unsqueeze(1)
        outputs['out_0'] = self.pyramid_0(x_latent).squeeze(-1)
        outputs['cls_0'] = self.cls_infer0(x_cls.squeeze(1)).squeeze(-1)

        x_latent = self.latent_compression_1(x_latent)
        # print(x_latent.shape)
        if BM_map != None:
            BM_map = self.BM_compression_1(BM_map)
            x_latent = self.to_bnc1(x_latent)
            # print(x_latent.shape)
            BM_map = self.to_bnc1(BM_map)
            x = torch.cat((x_cls, x_latent), dim=1)
            x = x + self.self_attn_1(x)
            x = x + self.cross_attn_1(x, BM_map)
            BM_map = self.to_bcw1(BM_map)
            x_cls = x[:, 0].unsqueeze(1)
            x_latent = self.to_bcw1(x[:, 1:])
            # print(x_latent.shape)

        outputs['out_1'] = self.pyramid_1(x_latent).squeeze(-1)
        outputs['cls_1'] = self.cls_infer1(x_cls.squeeze(1)).squeeze(-1)


        x_latent = self.latent_compression_2(x_latent)
        # print(x_latent.shape)
        if BM_map != None:
            BM_map = self.BM_compression_2(BM_map)
            x_latent = self.to_bnc2(x_latent)
            BM_map = self.to_bnc2(BM_map)
            # print(x_latent.shape)
            x = torch.cat((x_cls, x_latent), dim=1)
            x = x + self.self_attn_2(x)
            x = x + self.cross_attn_2(x, BM_map)
            BM_map = self.to_bcw2(BM_map)
            x_cls = x[:, 0].unsqueeze(1)
            x_latent = self.to_bcw2(x[:, 1:])
            # print(x_latent.shape)
        outputs['out_2'] = self.pyramid_2(x_latent).squeeze(-1)
        outputs['cls_2'] = self.cls_infer2(x_cls.squeeze(1)).squeeze(-1)

        
        return outputs



class FF_ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., FFC_to_Patch=True, shift=True, self_locality=True, cross_locality=True):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        print("H:{} X W:{} patches for ViT".format((image_height // patch_height), (image_width // patch_width)))
        patch_dim = channels * patch_height * patch_width

        if FFC_to_Patch:
            self.to_patch_embedding = FFC_patch_embeding(f_height=image_height // patch_height, f_width=image_width // patch_width, 
                                                         patch_height=patch_height, patch_width=patch_width, 
                                                         dim=dim, shift=shift)
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, self_locality=self_locality)
        self.to_latent = nn.Identity()
        self.pyr_infer = Pyramid_Angle_Infer(H=image_height // patch_height, W=image_width // patch_width, dim=dim, 
                                             self_locality=self_locality, cross_locality=cross_locality)

        

    def forward(self, img, BM_map=None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        out = self.pyr_infer(x, BM_map)

        return out
