import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from timm.models.layers import DropPath, Mlp, ConvMlp


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

class MyMaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        window_size: int = 2,
        index: int = 0,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.window_size = window_size

        self.q = nn.Linear(dim, dim_out)
        self.kv = nn.Linear(dim, 2 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)
        self.heads = 1 * (2 ** index)
        self.head_dim = self.dim_out // self.heads

        self.use_mask_unit_attn = [True, True, True, False][index]


    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
        B, N, _ = q.shape
        num_windows = (
            (N // self.window_size if self.use_mask_unit_attn else 1)
        )

        q = (self.q(q)
            .reshape(B, -1, num_windows, 1, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )

        kv = (
            self.kv(kv)
            .reshape(B, -1, num_windows, 2, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )

        q = q[0]
        k, v = kv[0], kv[1]

        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1)
            x = (attn @ v)

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x

class Conv3x3(nn.Module):
    """ MLP using 3x3 convs that keeps spatial dims
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=3, stride=1, padding=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class SPCV1(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        self.ln = nn.LayerNorm(dim)

        self.mlp = Mlp(dim, hdim, hdim)
        self.classifier = ConvMlp(hdim, hdim, odim)

    def forward(self, x):
        x = self.ln(x)
        x = self.mlp(x)
        classification = self.classifier(x.permute(0, 3, 1, 2))

        return x.permute(0, 3, 1, 2), classification


class SPCV2(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        self.ln = nn.LayerNorm(dim)

        self.mlp = ConvMlp(dim, hdim, hdim)
        self.classifier = ConvMlp(hdim, hdim, odim)

    def forward(self, x):
        x = self.ln(x)
        x = self.mlp(x.permute(0, 3, 1, 2))
        classification = self.classifier(x)

        return x, classification

class SPCV3(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        self.ln = nn.LayerNorm(dim)

        self.mlp = Mlp(dim, hdim, hdim)
        self.classifier = Mlp(hdim, hdim, odim)

    def forward(self, x):
        x = self.ln(x)
        x = self.mlp(x)
        classification = self.classifier(x)

        return x.permute(0, 3, 1, 2), classification.permute(0, 3, 1, 2)


class DPCV1(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        self.lnx = nn.LayerNorm(dim)
        self.lnres = nn.LayerNorm(dim)

        self.ln = nn.LayerNorm(dim)

        self.mlp = Mlp(dim * 2, hdim, hdim)
        self.classifier = Mlp(hdim, hdim, odim)

        self.res_up = nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x, res):

        res = self.res_up(res).permute(0, 2, 3, 1)

        x = self.lnx(x)
        res = self.lnres(res)

        xres = torch.cat([x, res], dim=-1)

        x = self.mlp(xres)
        classification = self.classifier(x)

        return x.permute(0, 3, 1, 2), classification.permute(0, 3, 1, 2)


class DPCV2(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        self.lnx = nn.LayerNorm(dim)
        self.lnres = nn.LayerNorm(dim)

        self.ln = nn.LayerNorm(dim * 2)

        self.mlp = Conv3x3(dim * 2, hdim, hdim)
        self.classifier = Mlp(hdim, hdim, odim)

        self.res_up = nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x, res):

        res = self.res_up(res).permute(0, 2, 3, 1)

        x = self.lnx(x)
        res = self.lnres(res)

        xres = torch.cat([x, res], dim=-1)

        xres = self.ln(xres)

        x = self.mlp(xres.permute(0, 3, 1, 2))
        classification = self.classifier(x.permute(0, 2, 3, 1))

        return x, classification.permute(0, 3, 1, 2)


class DPCV3(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        self.lnx = nn.LayerNorm(dim)
        self.lnres = nn.LayerNorm(dim)

        self.ln = nn.LayerNorm(dim * 2)

        self.conv_x = Conv3x3(dim, hdim, hdim)
        self.conv_res = Conv3x3(dim, hdim, hdim)

        self.classifier = Mlp(hdim * 2, hdim, odim)

        self.res_up = nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x, res):

        res = self.res_up(res).permute(0, 2, 3, 1)

        x = self.lnx(x)
        res = self.lnres(res)

        x = self.conv_x(x.permute(0, 3, 1, 2))
        res = self.conv_res(res.permute(0, 3, 1, 2))

        xres = torch.cat([x, res], dim=1).permute(0, 2, 3, 1)

        self.ln(xres)

        classification = self.classifier(xres)

        return x, classification.permute(0, 3, 1, 2)


class DPCV4(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        self.lnx = nn.LayerNorm(dim)
        self.lnres = nn.LayerNorm(dim)

        self.ln = nn.LayerNorm(dim)

        self.conv_xres = Conv3x3(dim, hdim, hdim)
        self.checkerboard_down = nn.Conv2d(dim, hdim, kernel_size=3, stride=2, padding=1)

        self.classifier = Mlp(hdim, hdim, odim)

        self.res_up = nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x, res):

        res = self.res_up(res).permute(0, 2, 3, 1)

        x = self.lnx(x).permute(0, 3, 1, 2)
        res = self.lnres(res).permute(0, 3, 1, 2)

        xres = checkerboard_tensors(x, res)
        xres = self.checkerboard_down(xres)
        xres = self.conv_xres(xres)

        x = xres

        xres = self.ln(xres.permute(0, 2, 3, 1))

        classification = self.classifier(xres)

        return x, classification.permute(0, 3, 1, 2)


class S1(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        
        self.ln = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hdim, hdim)
        self.cls = Mlp(hdim, hdim, odim)

    def forward(self, x):
        x = self.ln(x)
        x = self.mlp(x)
        cls = self.cls(x)

        return x.permute(0, 3, 1, 2), cls.permute(0, 3, 1, 2)


class S2(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        self.ln = nn.LayerNorm(dim)
        self.conv = Conv3x3(dim, hdim, hdim)
        self.cls = Mlp(hdim, hdim, odim)

    def forward(self, x):
        x = self.ln(x)
        x = self.conv(x.permute(0, 3, 1, 2))
        cls = self.cls(x.permute(0, 2, 3, 1))

        return x, cls.permute(0, 3, 1, 2)


class S3(nn.Module):
    def __init__(self, dim, hdim, odim, index=None):
        super().__init__()

        self.dim = dim
        self.hdim = hdim
        self.odim = odim

        self.classifier = nn.Conv2d(self.hdim, self.odim, kernel_size=3, stride=2, padding=1)

        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.ln = nn.LayerNorm(self.dim)
        self.conv = nn.Conv2d(self.dim * 2, self.dim, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.ln(x)

        b, h, w, c = x.shape

        x = x.permute(0, 3, 1, 2)

        xpool = self.avg_pool(x)
        maxpool = self.max_pool(x)

        pooled = checkerboard_tensors(xpool, maxpool)

        x = torch.cat([x, pooled], dim=1)

        x = self.conv(x)

        classification = self.classifier(x)

        return x, classification




class D1(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        self.ln = nn.LayerNorm(dim)

        self.res_up = nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1)

        self.mlp = Mlp(dim * 2, hdim * 2, hdim * 2)
        self.cls = Mlp(hdim * 2, hdim, odim)

    def forward(self, x, res):
        """

        :param x: should be of shape (b, c, h, w)
        :param res: should be of shape (b, c, h, w)
        :return: x (b, c, h, w), cls (b, h, w, odim)
        """

        res = self.res_up(res)

        x = self.ln(x)
        res = self.ln(res.permute(0, 2, 3, 1))

        xres = torch.cat([res, x], dim=-1)

        x = self.mlp(xres)
        cls = self.cls(x)

        return x.permute(0, 3, 1, 2), cls.permute(0, 3, 1, 2)


class D2(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        self.ln = nn.LayerNorm(dim)

        self.res_up = nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1)

        self.conv = Conv3x3(dim * 2, hdim, hdim)
        self.cls = Mlp(hdim, hdim, odim)

    def forward(self, x, res):
        """

        :param x: should be of shape (b, c, h, w)
        :param res: should be of shape (b, c, h, w)
        :return: x (b, c, h, w), cls (b, h, w, odim)
        """

        res = self.res_up(res)

        x = self.ln(x)
        res = self.ln(res.permute(0, 2, 3, 1))

        xres = torch.cat([res, x], dim=-1).permute(0, 3, 1, 2)

        x = self.conv(xres)
        cls = self.cls(x.permute(0, 2, 3, 1))

        return x, cls.permute(0, 3, 1, 2)


class D3(nn.Module):
    def __init__(self, dim, hdim, odim):
        super().__init__()

        self.ln = nn.LayerNorm(dim)

        self.res_up = nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1)

        self.x_conv = Conv3x3(dim, hdim, hdim)

        self.res_conv = Conv3x3(dim, hdim, hdim)

        self.cls = Mlp(hdim * 2, hdim, odim)

    def forward(self, x, res):
        """

        :param x: should be of shape (b, c, h, w)
        :param res: should be of shape (b, c, h, w)
        :return: x (b, c, h, w), cls (b, h, w, odim)
        """

        res = self.res_up(res)

        x = self.ln(x).permute(0, 3, 1, 2)
        res = self.ln(res.permute(0, 2, 3, 1)).permute(0, 3, 2, 1)

        x = self.x_conv(x)
        res = self.res_conv(res)

        xres = torch.cat([res, x], dim=1)

        cls = self.cls(xres.permute(0, 2, 3, 1))

        return x, cls.permute(0, 3, 1, 2)

class D4(nn.Module):
    def __init__(self, dim, hdim, odim, index=None):
        super().__init__()

        self.dim = dim
        self.hdim = hdim
        self.odim = odim

        self.classifier = nn.Conv2d(self.hdim, self.odim, kernel_size=3, stride=2, padding=1)

        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.ln = nn.LayerNorm(self.dim)
        self.conv = nn.Conv2d(self.dim * 2, self.dim, kernel_size=3, stride=1, padding=1)
        self.conv_res = nn.Conv2d(self.dim * 2, self.dim * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x, res):
        x = self.ln(x)
        res = self.ln(res.permute(0, 2, 3, 1))

        b, h, w, c = x.shape

        x = x.permute(0, 3, 1, 2)
        res = res.permute(0, 3, 1, 2)

        avgpool = self.avg_pool(res)
        maxpool = self.max_pool(x)

        pooled = checkerboard_tensors(avgpool, maxpool)

        x = torch.cat([x, pooled], dim=1)

        x_res = self.conv_res(x)
        x = self.conv(x)

        classification = self.classifier(x)

        return x, x_res, classification


class New(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.merge = nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(2 * dim)
        self.gelu = nn.GELU()

        self.res = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.cur_cls = nn.Conv2d(dim, num_classes, kernel_size=3, stride=1, padding=1)
        self.next_cls = nn.Conv2d(dim * 2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, res):

        assert x.shape == res.shape, f"x ({x.shape}) and res ({res.shape}) must have the same shape"

        b, n, c = x.shape

        h, w = int(n ** 0.5), int(n ** 0.5)

        xres = torch.cat([x, res], dim=-1)
        xres = self.norm1(xres)

        xres = xres.reshape(b, -1, h, w)

        merged = self.merge(xres)
        merged = self.gelu(merged)

        # res branch
        res = self.res(merged)
        res = self.gelu(res)

        # x branch
        cur_cls = self.cur_cls(merged)

        next_cls = self.next_cls(res)
        next_cls = next_cls

        # Merged should go to "outs"
        # Res should go to the next stage (and patch prediction stage)
        # Cls should go to the patch prediction stage

        res = res.flatten(2).permute(0, 2, 1)

        return merged, res, cur_cls, next_cls


class New2(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.merge = nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(2 * dim)
        self.gelu = nn.GELU()

        self.res = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.cur_cls = nn.Conv2d(dim, num_classes, kernel_size=3, stride=1, padding=1)
        # self.next_cls = nn.Linear(dim * 2, num_classes)

    def forward(self, x, res):

        assert x.shape == res.shape, f"x ({x.shape}) and res ({res.shape}) must have the same shape"

        b, n, c = x.shape

        h, w = int(n ** 0.5), int(n ** 0.5)

        xres = torch.cat([x, res], dim=-1)
        xres = self.norm1(xres)

        xres = xres.reshape(b, -1, h, w)

        merged = self.merge(xres)
        merged = self.gelu(merged)

        # res branch
        res = self.res(merged)
        res = self.gelu(res)

        # x branch
        cur_cls = self.cur_cls(merged)

        #next_cls = self.next_cls(res.permute(0, 2, 3, 1))
        #next_cls = next_cls.reshape(b, -1, h // 2, w // 2)

        # Merged should go to "outs"
        # Res should go to the next stage (and patch prediction stage)
        # Cls should go to the patch prediction stage

        res = res.flatten(2).permute(0, 2, 1)

        return merged, res, cur_cls, None


class NewSplit(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.x_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.res_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.x_norm = nn.LayerNorm(dim)
        self.res_norm = nn.LayerNorm(dim)
        self.gelu = nn.GELU()

        self.x = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.xres = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        self.cur_cls = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.next_cls = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        #self.cur_cls = nn.Conv2d(dim, num_classes, kernel_size=3, stride=1, padding=1)
        #self.next_cls = nn.Conv2d(dim * 2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, res):

        assert x.shape == res.shape, f"x ({x.shape}) and res ({res.shape}) must have the same shape"

        b, n, c = x.shape

        h, w = int(n ** 0.5), int(n ** 0.5)

        x = self.x_norm(x).reshape(b, -1, h, w)
        res = self.res_norm(res).reshape(b, -1, h, w)

        # x : b, c, h, w
        # res: b, c, h, w

        x = self.gelu(self.x_conv(x))
        res = self.gelu(self.res_conv(res))

        xres = torch.cat([x, res], dim=1)

        x = self.x(x)
        xres = self.xres(xres)

        # x branch
        cur_cls = self.cur_cls(xres)
        next_cls = self.next_cls(x)

        # xres should go to "outs"
        # x should go to the next stage (and patch prediction stage)
        # Cls should go to the patch prediction stage

        xres = xres.flatten(2).permute(0, 2, 1)
        x = x.flatten(2).permute(0, 2, 1)

        return xres, x, cur_cls, next_cls


class NewSplit2(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.x_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.res_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.x_norm = nn.LayerNorm(dim)
        self.res_norm = nn.LayerNorm(dim)
        self.gelu = nn.GELU()

        self.x = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.xres = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        self.cur_cls = nn.Conv2d(dim, num_classes, kernel_size=3, stride=1, padding=1)
        # self.next_cls = nn.Conv2d(dim * 2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, res):

        assert x.shape == res.shape, f"x ({x.shape}) and res ({res.shape}) must have the same shape"

        b, n, c = x.shape

        h, w = int(n ** 0.5), int(n ** 0.5)

        x = self.x_norm(x).reshape(b, -1, h, w)
        res = self.res_norm(res).reshape(b, -1, h, w)

        # x : b, c, h, w
        # res: b, c, h, w

        x = self.gelu(self.x_conv(x))
        res = self.gelu(self.res_conv(res))

        xres = torch.cat([x, res], dim=1)

        x = self.x(x)
        xres = self.xres(xres)

        # x branch
        cur_cls = self.cur_cls(xres)

        # next_cls = self.next_cls(x)

        # xres should go to "outs"
        # x should go to the next stage (and patch prediction stage)
        # Cls should go to the patch prediction stage

        xres = xres.flatten(2).permute(0, 2, 1)
        x = x.flatten(2).permute(0, 2, 1)

        return xres, x, cur_cls, None

class NewSmall(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.x_norm = nn.LayerNorm(dim)
        self.res_norm = nn.LayerNorm(dim)
        self.gelu = nn.GELU()

        self.down = nn.Linear(dim, dim // 2)

        self.cls = nn.Linear(dim, num_classes)
        # self.next_cls = nn.Linear(dim * 2, num_classes)

        self.res = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

        self.out_norm = nn.LayerNorm(dim)


    def forward(self, x, res):

        assert x.shape == res.shape, f"x ({x.shape}) and res ({res.shape}) must have the same shape"

        b, n, c = x.shape

        h, w = int(n ** 0.5), int(n ** 0.5)

        x = self.x_norm(x).reshape(b, h, w, -1)
        res = self.res_norm(res).reshape(b, h, w, -1)

        x = self.down(x)
        res = self.down(res)

        x = self.gelu(x)
        res = self.gelu(res)

        catted = torch.cat([x, res], dim=-1)

        cur_cls = self.cls(catted)

        catted = catted.permute(0, 3, 1, 2)

        res = self.res(catted)
        res = self.gelu(res)
        # next_cls = self.next_cls(res.permute(0, 2, 3, 1))

        catted = catted.flatten(2).permute(0, 2, 1)
        res = res.flatten(2).permute(0, 2, 1)

        cur_cls = cur_cls.permute(0, 3, 1, 2)
        # next_cls = next_cls.permute(0, 3, 1, 2)

        catted = self.out_norm(catted)

        return catted, res, cur_cls, None

class PatchResidual(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.dim = dim
        self.num_classes = num_classes

        self.xnorm = nn.BatchNorm2d(dim)
        self.resnorm = nn.BatchNorm2d(dim)
        self.gelu = nn.GELU()

        self.cur_cls = nn.Conv2d(self.dim * 2, self.num_classes, kernel_size=1, stride=1, padding=0)

        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.dim_up = nn.Conv2d(self.dim, self.dim * 2, kernel_size=1, stride=1)

    def forward(self, x, res):

        assert x.shape == res.shape, f"x ({x.shape}) and res ({res.shape}) must have the same shape"

        b, n, c = x.shape

        h, w = int(n ** 0.5), int(n ** 0.5)

        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        res = res.reshape(b, h, w, -1).permute(0, 3, 1, 2)

        x = self.xnorm(x)
        res = self.resnorm(res)

        catted = torch.cat([x, res], dim=1)

        cur_cls = self.cur_cls(catted)

        xres = self.max_pool(catted)

        xres = xres.flatten(2).permute(0, 2, 1)

        return catted, xres, cur_cls, None

class PatchNonResidual(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.dim = dim
        self.num_classes = num_classes

        self.norm = nn.BatchNorm2d(dim)
        self.gelu = nn.GELU()

        self.mlp = nn.Linear(self.dim, self.dim)
        self.cur_cls = nn.Conv2d(self.dim, self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x, res):

        b, n, c = x.shape

        h, w = int(n ** 0.5), int(n ** 0.5)

        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)

        x = self.norm(x)

        cur_cls = self.cur_cls(x)

        return None, None, cur_cls, None


class PatchClassifier(nn.Module):
    def __init__(self, encoder_channels, num_patch_classes):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_classes = num_patch_classes

        self.patch_classifiers = nn.ModuleList([PatchResidual(enc, self.num_classes) for enc in self.encoder_channels])

    def forward(self, x, res, idx):
        return self.patch_classifiers[idx](x, res)

class DualPatchClassifier(nn.Module):
    def __init__(self, encoder_channels, num_patch_classes):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_classes = num_patch_classes

        self.patch_classifiers = nn.ModuleList([S2(enc, enc, self.num_classes) if self.encoder_channels.index(enc) == 0 else D3(enc, enc, self.num_classes) for enc in self.encoder_channels])

    def forward(self, x, res, idx):

        if idx == 0:
            return self.patch_classifiers[idx](x)

        return self.patch_classifiers[idx](x, res)
