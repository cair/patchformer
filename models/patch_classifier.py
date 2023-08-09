import torch
import torch.nn as nn
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



class PatchClassifier(nn.Module):
    def __init__(self, encoder_channels, num_patch_classes):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_classes = num_patch_classes
        self.patch_classifiers = nn.ModuleList([S1(enc, enc, self.num_classes) for enc in self.encoder_channels])

    def forward(self, x, idx):
        return self.patch_classifiers[idx](x)

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
