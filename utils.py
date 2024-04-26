from dataclasses import dataclass
from importlib.resources import path
import torch 
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from einops import rearrange


class GDER(nn.Module):
    def __init__(self):
        super(GDER, self).__init__()
        N = 15 // 2
        sig = N / 2.5
        a = np.arange(-N, N+1)
        x, y = np.meshgrid(a, a)
        G = np.exp(-(x**2 + y**2) / (2 * sig**2)) / (2*np.pi*sig)
        Gx = -x * G / (sig**2)
        self.Gx = torch.from_numpy(Gx / (np.sum(Gx)+1e-7)).view(1, 1, 15, 15).float()
        Gy = -y * G / (sig**2)
        self.Gy = torch.from_numpy(Gy / (np.sum(Gy))+1e-7).view(1, 1, 15, 15).float()
    
    def forward(self, x):
        Rx = F.conv2d(x, self.Gx.to(x.device))
        Ry = F.conv2d(x, self.Gy.to(x.device))
        FM = Rx**2 + Ry**2

        return torch.mean(FM, dim=(-1,-2))



class BM(nn.Module):
    def __init__(self):
        super(BM,self).__init__()
        self.aver_h = nn.AvgPool2d((1,9), 1)
        self.aver_v = nn.AvgPool2d((9,1), 1)
        self.pad_h = nn.ZeroPad2d((4, 4, 0, 0))
        self.pad_v = nn.ZeroPad2d((0, 0, 4, 4))
    
    def forward(self, x):
        x_h = self.pad_h(x)
        x_v = self.pad_v(x)

        B_hor = self.aver_h(x_h)
        B_ver = self.aver_v(x_v)

        D_F_ver = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        D_F_hor = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])

        D_B_ver = torch.abs(B_ver[:, :, :-1, :] - B_ver[:, :, 1:, :])
        D_B_hor = torch.abs(B_hor[:, :, :, :-1] - B_hor[:, :, :, 1:])

        T_ver = D_F_ver - D_B_ver
        T_hor = D_F_hor - D_B_hor

        V_ver = torch.maximum(T_ver, torch.tensor([0]).to(x.device))
        V_hor = torch.maximum(T_hor, torch.tensor([0]).to(x.device))

        S_V_ver = torch.sum(V_ver[:, :, 1:-1, 1:-1], dim=(-2, -1))
        S_V_hor = torch.sum(V_hor[:, :, 1:-1, 1:-1], dim=(-2, -1))
        
        blur = torch.maximum(S_V_ver, S_V_hor)
        return blur

def norm(x):
    I_max = torch.amax(x, dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1)
    I_min = torch.amin(x, dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1)
    x = (x - I_min) / (I_max - I_min)

    return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, BM):
        """
        :param tensor: A 3d tensor of size (batch_size, x, 1)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(BM.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")


        self.cached_penc = None
        sin_inp_x = BM * self.inv_freq.to(BM.device)
        self.cached_penc = get_emb(sin_inp_x)
        
        return self.cached_penc

def generate_BM_map(x, H_patch=32, W_patch=55, metric=None, embedding=None):
    
    b, _, h, w = x.size()
    h_steps = h // H_patch
    w_steps = w // W_patch

    map = torch.zeros(b, 1, h_steps, w_steps).to(x.device)
    for i in range(h_steps):
        for j in range(w_steps):
            map[:, :, i, j] = metric(x[:, :, i*H_patch:(i+1)*H_patch, j*W_patch:(j+1)*W_patch])
    
    map = norm(map)
    map = rearrange(map, 'b c h w -> b (h w) c')
    map_embedded = embedding(map)

    return map_embedded

