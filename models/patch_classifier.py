import torch
import torch.nn as nn

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class FCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.3):
        super(FCN, self).__init__()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_features, hidden_features // 2, kernel_size=3, padding=1)
        self.mlp = Mlp(hidden_features // 2, hidden_features // 2, out_features, act_layer, drop)

    def forward(self, x):
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.act(self.conv2(x)))
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)
        return x

class PatchClassifier(nn.Module):
    def __init__(self, encoder_channels, num_classes, binary=False):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_classes = num_classes
        self.binary = binary
        if self.binary:
            self.patch_classifiers = nn.ModuleList([FCN(enc, enc, 1) for enc in self.encoder_channels])
        else:
            self.patch_classifiers = nn.ModuleList([FCN(enc, enc, self.num_classes) for enc in self.encoder_channels])

