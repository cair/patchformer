import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import einops

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class PatchExpansion(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class Introspection(nn.Module):
    def __init__(self, embed_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim

        self.ln = nn.LayerNorm(self.embed_dim)
        self.cls = nn.Conv2d(self.embed_dim, 2, kernel_size=1, stride=1)

    def forward(self, x):
        return self.cls(self.ln(x))

class Unfold(nn.Module):
    def __init__(self, embed_dim, window_size = 2, stride = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.reduction = nn.Conv2d(embed_dim * (window_size * stride), embed_dim * (window_size), kernel_size=1, stride=1)
        self.window_size = window_size
        self.stride = stride

    def forward(self, x):

        b, c, h, w = x.shape

        patches = x.unfold(2, self.window_size, self.stride).unfold(3, self.window_size, self.stride)

        reshaped = patches.contiguous().view(b, c * (self.window_size * self.stride), h // self.stride, w // self.stride) # Expects rectangular input

        reshaped = self.reduction(reshaped)

        return reshaped

class Fold(nn.Module):
    def __init__(self, embed_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.reduction = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, stride=1)

    def forward(self, x):
        pass

    


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), attn.permute(0, 2, 3, 1).mean(dim=-1).mean(dim=-1)
    

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            xtemp, attn_map  = attn(x)
            x = xtemp + x
            x = ff(x) + x

        return self.norm(x), attn_map


class NewFormer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.img_size = 256

        self.patch_sizes = [4, 8, 16, 32]
        self.embed_dims = [96, 192, 384, 768]
        self.heights = [self.img_size // patch_size for patch_size in self.patch_sizes]
        self.widths = [self.img_size // patch_size for patch_size in self.patch_sizes]

        self.depths = [4, 4, 4, 4]

        self.patch_dim = self.patch_sizes[0] * self.patch_sizes[0] * 3

        self.num_patches = self.img_size // self.patch_sizes[0] * self.img_size // self.patch_sizes[0]

        # Create internal positional encoding with smallest patch size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_sizes[0], p2=self.patch_sizes[0]),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.embed_dims[0]),
            nn.LayerNorm(self.embed_dims[0]),
        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dims[0]))

        unfold = "stepwise"

        if unfold == "direct":
            self.unfold = Unfold(self.embed_dims[0], window_size=8, stride=8)
        else:
            self.unfold8 = Unfold(self.embed_dims[0], window_size=2, stride=2)
            self.unfold16 = Unfold(self.embed_dims[1], window_size=2, stride=2)
            self.unfold32 = Unfold(self.embed_dims[2], window_size=2, stride=2)
        
        
        self.transformer32 = Transformer(self.embed_dims[-1], self.depths[-1], 16, 256, self.embed_dims[-1] * 4)
        self.transformer16 = Transformer(self.embed_dims[-2], self.depths[-2], 8, 128, self.embed_dims[-2] * 4)
        self.transformer8 = Transformer(self.embed_dims[-3], self.depths[-3], 4, 64, self.embed_dims[-3] * 4)
        self.transformer4 = Transformer(self.embed_dims[-4], self.depths[-4], 2, 32, self.embed_dims[-4] * 4)

        self.reduce32 = nn.Conv2d(self.embed_dims[-1], self.embed_dims[-2], kernel_size=1, stride=1)
        self.reduce16 = nn.Conv2d(self.embed_dims[-2], self.embed_dims[-3], kernel_size=1, stride=1)
        self.reduce8 = nn.Conv2d(self.embed_dims[-3], self.embed_dims[-4], kernel_size=1, stride=1)


    def get_top_indices(self, original_indices, attn_map, current_patch_size, target_patch_size):

        _, indices = torch.sort(attn_map, dim=1, descending=True)
        cutoff = int(0.75 * attn_map.shape[1])

        top_indices = indices[:, :cutoff]

        B, I = top_indices.shape

        top_indices = torch.gather(original_indices, 1, top_indices)

        top_indices = self.get_indices_from_smaller_patches_batch(top_indices, self.img_size, current_patch_size, target_patch_size)

        return top_indices

    def get_indices_from_smaller_patches_batch(self, indices, S, window_size, target_window_size):
        """
        Given a batch of indices from the specified window_size, returns the corresponding indices from the target_window_size version.
        :param indices: Tensor of shape [B, I]
        :param S: Image size (S x S)
        :param window_size: Current window size (e.g., 32 for 32x32 patches)
        :param target_window_size: Target window size (e.g., 16 for 16x16 patches)
        :return: Tensor of shape [B, I, 4]
        """
        B, I = indices.shape

        # Calculate the rows and columns for the given indices in the current window_size
        rows_current = indices // (S // window_size)
        cols_current = indices % (S // window_size)

        # Calculate the scaling factor between the window sizes
        scaling_factor = window_size // target_window_size

        # Calculate the starting rows and columns for the target_window_size
        start_rows_target = rows_current * scaling_factor
        start_cols_target = cols_current * scaling_factor

        # Calculate the indices in the target_window_size
        index1 = start_rows_target * (S // target_window_size) + start_cols_target
        index2 = index1 + 1
        index3 = (start_rows_target + 1) * (S // target_window_size) + start_cols_target
        index4 = index3 + 1

        indices_target = torch.stack([index1, index2, index3, index4], dim=-1)

        return indices_target

    def get_indices_from_larger_patches_batch(self, indices, S, window_size, target_window_size):
        """
        Given a batch of sets of indices from the smaller target_window_size, returns the corresponding indices from the window_size version.
        :param indices: Tensor of shape [B, I, 4] where I is the number of sets and 4 represents indices from target_window_size
        :param S: Image size (S x S)
        :param window_size: Larger window size to get the index from
        :param target_window_size: Smaller window size the provided indices belong to
        :return: Tensor of shape [B, I]
        """

        # Extract the first index from each set; this is enough to determine the larger patch index
        first_indices = indices[:, :, 0]

        # Calculate the rows and columns for the given first indices in the target_window_size
        rows_target = first_indices // (S // target_window_size)
        cols_target = first_indices % (S // target_window_size)

        # Calculate the scaling factor between the window sizes
        scaling_factor = window_size // target_window_size

        # Calculate the rows and columns for the larger window size
        rows_larger = rows_target // scaling_factor
        cols_larger = cols_target // scaling_factor

        # Calculate the indices in the larger window size
        indices_larger = rows_larger * (S // window_size) + cols_larger

        return indices_larger

    def forward(self, x) -> None:
        
        # First create patches with the smallest patch size
        x = self.to_patch_embedding(x) + self.pos_embedding

        x = einops.rearrange(x, "b (h w) c -> b c h w", h=self.heights[0], w=self.widths[0])

        x8_original = self.unfold8(x) # double channels and half height and width
        x16_original = self.unfold16(x8_original) # double channels and half height and width
        x32_original = self.unfold32(x16_original) # double channels and half height and width

        x32 = x32_original.flatten(2).permute(0, 2, 1) # b c h w -> b n c

        x32, attn_map = self.transformer32(x32)

        x32_original_indices = torch.arange(0, attn_map.shape[1]).repeat(attn_map.shape[0], 1).to(attn_map.device)

        x16_original_indices = self.get_top_indices(x32_original_indices, attn_map, 32, 16).flatten(1)

        x16 = x16_original.flatten(2).permute(0, 2, 1)
        x16 = torch.gather(x16, 1, x16_original_indices.unsqueeze(-1).expand(-1, -1, x16.size(2)))

        x16, attn_map = self.transformer16(x16)

        x8_original_indices = self.get_top_indices(x16_original_indices, attn_map, 16, 8).flatten(1)

        x8 = x8_original.flatten(2).permute(0, 2, 1)
        x8 = torch.gather(x8, 1, x8_original_indices.unsqueeze(-1).expand(-1, -1, x8.size(2)))

        x8, attn_map = self.transformer8(x8)

        x4_original_indices = self.get_top_indices(x8_original_indices, attn_map, 8, 4).flatten(1)

        x4 = x.flatten(2).permute(0, 2, 1)
        x4 = torch.gather(x4, 1, x4_original_indices.unsqueeze(-1).expand(-1, -1, x4.size(2)))

        x4, attn_map = self.transformer4(x4)

        # interpolate from x32 to x16
        b, n, c = x32.shape
        x16_interp = F.interpolate(self.reduce32(x32.reshape(b, c, self.heights[-1], self.widths[-1])), scale_factor=2).flatten(2).permute(0, 2, 1)

        x16_interp[:, x16_original_indices[:]] = x16

        b, n, c = x16_interp.shape
        x8_interp = F.interpolate(self.reduce16(x16_interp.reshape(b, c, self.heights[-2], self.widths[-2])), scale_factor=2).flatten(2).permute(0, 2, 1)

        x8_interp[:, x8_original_indices[:]] = x8

        b, n, c = x8_interp.shape
        x4_interp = F.interpolate(self.reduce8(x8_interp.reshape(b, c, self.heights[-3], self.widths[-3])), scale_factor=2).flatten(2).permute(0, 2, 1)

        x4_interp[:, x4_original_indices[:]] = x4


        x32 = x32.reshape(-1, self.embed_dims[-1], self.heights[-1], self.widths[-1])
        x16 = x16_interp.reshape(-1, self.embed_dims[-2], self.heights[-2], self.widths[-2])
        x8 = x8_interp.reshape(-1, self.embed_dims[-3], self.heights[-3], self.widths[-3])
        x4 = x4_interp.reshape(-1, self.embed_dims[-4], self.heights[-4], self.widths[-4])

        return [x4, x8, x16, x32]