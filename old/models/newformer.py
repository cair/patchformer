import einops
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from models.dcswin import Decoder

import math


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

def checkerboard_tensors(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"
    b, c, h, w = tensor1.shape

    # Ensure the height and width are even
    assert h % 2 == 0 and w % 2 == 0, "Height and width must be even numbers"

    # Reshape each tensor to have a dimension for rows and cols
    tensor1 = tensor1.reshape(b, c, h, 1, w, 1)
    tensor2 = tensor2.reshape(b, c, h, 1, w, 1)

    # Create two kinds of interleaved patterns and stack them alternatively
    pattern1 = torch.stack((tensor1, tensor2), dim=-1)
    pattern2 = torch.stack((tensor2, tensor1), dim=-1)

    pattern = torch.stack((pattern1, pattern2), dim=4)

    # Interleaving across height and width dimensions
    checkerboard = pattern.reshape(b, c, h * 2, w * 2)

    return checkerboard

def patchify_mask(mask, patch_size):
    # Input mask is of size (b, h, w)
    # Get batch size, height and width
    b, h, w = mask.shape

    # Create patches using unfold. The resulting size is (b, h', w', patch_size, patch_size)
    # where h' and w' are the number of patches in height and width dimension respectively
    patches = mask.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

    # Now we have to change the shape to (b, h', w', patch_size*patch_size)
    patched_mask = patches.contiguous().view(b, h // patch_size, w // patch_size, patch_size * patch_size)

    return patched_mask

def check_heterogeneity_binary(tensor, ignore_index):
    # Get the shape of the tensor
    original_shape = tensor.shape
    last_dim = original_shape[-1]

    # Reshape the tensor to 2D for easy comparison
    tensor = tensor.reshape(-1, last_dim)

    # Create a mask for elements to ignore
    ignore_mask = tensor.eq(ignore_index)

    # Compare each element with the first along the last dimension, ignoring specified class
    tensor_bool = tensor.eq(tensor[:, 0].unsqueeze(1)) & ~ignore_mask

    # Check if all elements along the last dimension are the same, ignoring specified class
    tensor_bool = tensor_bool.all(dim=-1)

    # Negate the boolean tensor to get heterogeneity
    tensor_bool = ~tensor_bool

    # Reshape the tensor back to original shape (minus last dimension)
    tensor_bool = tensor_bool.reshape(original_shape[:-1])

    return tensor_bool.long()

def reshape_tensor(tensor):
    b, hw, c = tensor.shape

    h = int(math.sqrt(hw))
    w = h
    tensor = tensor.reshape(b, c, h, w)

    return tensor

class NewFormerDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.patch_sizes = [32, 16, 8, 4]

        self.up1 = nn.ConvTranspose2d(in_channels=768, out_channels=384, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=3, stride=2, padding=1, output_padding=1)

        #self.attn1 = Transformer(384, 1, 4, 64, 384)
        #self.attn2 = Transformer(192, 1, 4, 64, 192)
        #self.attn3 = Transformer(96, 1, 4, 64, 96)

        self.mix1 = nn.Linear(384 * 2, 384)
        self.mix2 = nn.Linear(192 * 2, 192)
        self.mix3 = nn.Linear(96 * 2, 96)

        self.pre_classifier = nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.classifier = nn.ConvTranspose2d(in_channels=96, out_channels=num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.gelu = nn.GELU()

    def reshape_tensor(self, tensor):
        b, hw, c = tensor.shape

        h = int(math.sqrt(hw))
        w = h
        tensor = tensor.view(b, c, h, w)

        return tensor

    def forward(self, x1, x2, x3, x4):
        # Reshape all to fit convolutions
        x1 = self.reshape_tensor(x1)

        x = self.gelu(self.up1(x1))

        if x2 is not None:
            x2 = self.reshape_tensor(x2)
            x = self.mix1(torch.cat([x, x2], dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            #x = self.attn1(x.flatten(2, 3).permute(0, 2, 1))
            #x = self.reshape_tensor(x)

        x = self.gelu(self.up2(x))

        if x3 is not None:
            x3 = self.reshape_tensor(x3)
            x = self.mix2(torch.cat([x, x3], dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            #x = self.attn2(x.flatten(2, 3).permute(0, 2, 1))
            #x = self.reshape_tensor(x)

        x = self.gelu(self.up3(x))

        if x4 is not None:
            x4 = self.reshape_tensor(x4)
            x = self.mix3(torch.cat([x, x4], dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            #x = self.attn3(x.flatten(2, 3).permute(0, 2, 1))
            #x = self.reshape_tensor(x)

        x = self.gelu(self.pre_classifier(x))
        x = self.classifier(x)

        return x

from typing import List, Tuple, Callable, Optional, Type, Dict, Any

class PatchEmbed(nn.Module):
    """Patch embed that supports any number of spatial dimensions (1d, 2d, 3d)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
    ):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x = do_masked_conv(x, self.proj, mask)
        x = self.proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x

class FormerBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_sizes = [2, 8, 32]
        self.embedding_dims = [48, 192, 768]
        self.patch_embed = PatchEmbed(3, 768, (self.patch_sizes[0], self.patch_sizes[0]), (self.patch_sizes[0], self.patch_sizes[0]), (0, 0))



class NewFormerBackbone(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.patch_sizes = [4, 8, 16, 32]
        self.patch_dims = [3 * self.patch_sizes[i] * self.patch_sizes[i] for i in range(len(self.patch_sizes))]
        self.dims = [96, 192, 384, 768]

        self.image_height, self.image_width = image_size

        self.patch_embed1 = self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = self.patch_sizes[0], p2 = self.patch_sizes[0]),
            nn.LayerNorm(self.patch_dims[0]),
            nn.Linear(self.patch_dims[0], self.dims[0]),
            nn.LayerNorm(self.dims[0]),
        )

        self.stage_1 = Transformer(self.dims[0], 2, 6, 64, self.dims[0])
        self.pos_embedding_1 = posemb_sincos_2d(
            h=self.image_height // self.patch_sizes[0],
            w=self.image_width // self.patch_sizes[0],
            dim=self.dims[0],
        )


        self.patch_embed2 = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_sizes[1], p2=self.patch_sizes[1]),
            nn.LayerNorm(self.patch_dims[1]),
            nn.Linear(self.patch_dims[1], self.dims[1]),
            nn.LayerNorm(self.dims[1]),
        )
        self.stage_2 = Transformer(self.dims[1], 4, 8, 64, self.dims[1])
        self.pos_embedding_2 = posemb_sincos_2d(
            h=self.image_height // self.patch_sizes[1],
            w=self.image_width // self.patch_sizes[1],
            dim=self.dims[1],
        )


        self.patch_embed3 = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_sizes[2], p2=self.patch_sizes[2]),
            nn.LayerNorm(self.patch_dims[2]),
            nn.Linear(self.patch_dims[2], self.dims[2]),
            nn.LayerNorm(self.dims[2]),
        )
        self.stage_3 = Transformer(self.dims[2], 6, 12, 64, self.dims[2])
        self.pos_embedding_3 = posemb_sincos_2d(
            h=self.image_height // self.patch_sizes[2],
            w=self.image_width // self.patch_sizes[2],
            dim=self.dims[2],
        )

        self.patch_embed4 = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_sizes[3], p2=self.patch_sizes[3]),
            nn.LayerNorm(self.patch_dims[3]),
            nn.Linear(self.patch_dims[3], self.dims[3]),
            nn.LayerNorm(self.dims[3]),
        )
        self.stage_4 = Transformer(self.dims[3], 8, 16, 64, self.dims[3])
        self.pos_embedding_4 = posemb_sincos_2d(
            h=self.image_height // self.patch_sizes[3],
            w=self.image_width // self.patch_sizes[3],
            dim=self.dims[3],
        )


    def forward(self, x, mask=None):

        x1 = self.patch_embed1(x)
        x1 += self.pos_embedding_1.to(x.device)
        x1 = self.stage_1(x1)

        x2 = self.patch_embed2(x)
        x2 += self.pos_embedding_2.to(x.device)
        x2 = self.stage_2(x2)

        x3 = self.patch_embed3(x)
        x3 += self.pos_embedding_3.to(x.device)
        x3 = self.stage_3(x3)

        x4 = self.patch_embed4(x)
        x4 += self.pos_embedding_4.to(x.device)
        x4 = self.stage_4(x4)

        return x1, x2, x3, x4


class NewFormer(nn.Module):
    def __init__(self, image_size, num_classes):
        super().__init__()

        self.backbone = NewFormerBackbone(image_size)
        #self.decoder = NewFormerDecoder(num_classes)
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x, mask=None):

        x1, x2, x3, x4 = self.backbone(x, mask)

        xs = [x1, x2, x3, x4]

        for idx, x in enumerate(xs):
            xs[idx] = reshape_tensor(x)

        x = self.decoder(*xs)

        return x, None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

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
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        """
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )
        """
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.LayerNorm(dim)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        # x = x.mean(dim = 1)
        # x = self.to_latent(x)

        return self.linear_head(x)


def new_tiny(input_size, num_classes):
    model = NewFormer(input_size, num_classes)

    return model

if __name__ == "__main__":

    model = NewFormer((512, 512), 2)

    x = torch.randn(3, 3, 512, 512)
    mask = torch.round(torch.rand(3, 512, 512))

    y = model(x, mask=mask)