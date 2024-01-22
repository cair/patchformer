import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ConvForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=3, stride=1, padding=1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class PatchClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.initial_norm = nn.LayerNorm(dim)

        self.gelu = nn.GELU()
        
        self.sem_mlp = nn.Linear(dim, dim)
        self.semantic_norm = nn.LayerNorm(dim)
        
        self.patch_cls = nn.Linear(dim, num_classes)
        
        self.mlp = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        
        x = x.permute(0, 2, 3, 1)
        
        x = self.initial_norm(x)

        semantic = self.sem_mlp(x)
        semantic = self.semantic_norm(semantic)
        semantic = self.gelu(semantic)
        
        cls = self.patch_cls(semantic)
        
        x = torch.cat([x, semantic], dim=-1)
        
        x = self.mlp(x)
        x = self.norm(x)
        x = self.gelu(x)
        
        x = x.permute(0, 3, 1, 2)
        cls = cls.permute(0, 3, 1, 2)
        
        return x, cls


class HiearchicalPatchClassifier(nn.Module):
    def __init__(self, dims, num_classes, cls_type="conv1x1"):
        super().__init__()

        self.initial_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in dims
        ])
        
        self.cls_type = cls_type
        
        if cls_type == "conv1x1":
            self.patch_cls = nn.Conv2d(sum(dims), num_classes, kernel_size=1, stride=1)
        elif cls_type == "conv3x3":
            self.patch_cls = nn.Conv2d(sum(dims), num_classes, kernel_size=3, stride=1, padding=1)
        elif cls_type == "mlp":
            self.patch_cls = nn.Linear(sum(dims), num_classes)
        
        
    def forward(self, x):
        
        orig_x = x
        
        for xdx, xi in enumerate(x):
            temp = self.initial_norms[xdx](xi.permute(0, 2, 3, 1))
            x[xdx] = temp.permute(0, 3, 1, 2)
            
        interpolate_candidates = reversed(x[1:])
        
        previous_candidate = None
        
        for cdx, candidate in enumerate(interpolate_candidates):
            if cdx > 0:
                candidate = torch.cat([candidate, previous_candidate], dim=1)
                previous_candidate = F.interpolate(candidate, scale_factor=2, mode="bilinear")
            else:
                previous_candidate = F.interpolate(candidate, scale_factor=2, mode="bilinear")
            
        x = torch.cat([x[0], previous_candidate], dim=1)
        
        if self.cls_type == "mlp":
            x = x.permute(0, 2, 3, 1)
        
        cls = self.patch_cls(x)
        
        if self.cls_type == "mlp":
            cls = cls.permute(0, 3, 1, 2)
        
        return orig_x, cls

class FlatPatchClassifier(nn.Module):
    def __init__(self, dims, num_classes, cls_type="conv1x1"):
        super().__init__()

        self.initial_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in dims
        ])
        
        self.cls_type = cls_type
        
        if cls_type == "conv1x1":
            self.patch_cls = nn.Conv2d(sum(dims), num_classes, kernel_size=1, stride=1)
        elif cls_type == "conv3x3":
            self.patch_cls = nn.Conv2d(sum(dims), num_classes, kernel_size=3, stride=1, padding=1)
        elif cls_type == "mlp":
            self.patch_cls = nn.Linear(sum(dims), num_classes)
        
    def forward(self, x):
        
        orig_x = x
        
        for xdx, xi in enumerate(x):
            temp = self.initial_norms[xdx](xi.permute(0, 2, 3, 1))
            x[xdx] = temp.permute(0, 3, 1, 2)
            
        x = torch.cat([*x], dim=1)
        
        if self.cls_type == "mlp":
            x = x.permute(0, 2, 3, 1)
            
        cls = self.patch_cls(x)
        
        if self.cls_type == "mlp":
            cls = cls.permute(0, 3, 1, 2)
        
        return orig_x, cls