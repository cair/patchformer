from collections import namedtuple
from packaging import version

import torch
import torch.nn.functional as F
from torch import nn


from einsum import einsum
from einops import rearrange
from einops.layers.torch import Rearrange

# constants

Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# main class

class Attend(nn.Module):
    def __init__(self, use_flash=False):
        super().__init__()
        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse(
            '2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = Config(True, False, False)
        else:
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v):
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # flash attention - https://arxiv.org/abs/2205.14135

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v)

        return out

    def forward(self, q, k, v):
        n, device, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, use_flash=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = Attend(use_flash=use_flash)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, q, kv):
        q = self.norm(q)
        kv = self.norm(kv)

        kv = self.to_kv(kv).chunk(2, dim=-1)
        q = self.to_q(q)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)