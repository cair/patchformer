import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm import create_model

from utils import iou_pytorch as iou, acc_pytorch as acc, patchify_mask, check_homogeneity_classes, check_homogeneity_majority, compute_class_proportions
from models.decoders.upernet import UPerNet
from models.patch_classifiers.patch_classifier import FlatPatchClassifier
from losses.dice import DiceLoss

import math
from statistics import mean

class ViTUperNet(L.LightningModule):
    def __init__(self,
                 num_classes: int,
                 learning_rate: float,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 patch_learning: bool = True,
                 patch_size: int = 16,
                 model_size: str = "small",
                 cls_type: str = "conv1x1"
                 ):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.vt = create_model(f"vit_{model_size}_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
        
        # if model_size == "base":
        #     self.vt = create_model(f"vit_{model_size}_patch16_384.augreg_in21k_ft_in1k", pretrained=True)
        # elif model_size == "small":
        #     self.vt = create_model(f"vit_{model_size}_patch16_384.augreg_in21k_ft_in1k", pretrained=True)
        # elif model_size == "tiny":
        #     self.vt = create_model(f"vit_{model_size}_patch16_384.augreg_in21k_ft_in1k", pretrained=True)
            
        
        self.dim = self.vt.num_features
        
        self.scale_factor = 4 if self.patch_size == 8 else 8
        
        self.dec = UPerNet(num_class=num_classes, 
                           fc_dim=self.dim, 
                           fpn_inplanes=(self.dim, self.dim, self.dim, self.dim), 
                           fpn_dim=512, 
                           scale_factor=self.scale_factor)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        self.loss = DiceLoss(mode="multiclass", ignore_index=self.num_classes)
        
        self.patch_learning = patch_learning
        
        # Training metrics
        self.train_iou = list()
        self.train_acc = list()
        self.train_loss = list()

        # Validation metrics
        self.val_iou = list()
        self.val_acc = list()
        self.val_loss = list()
        
        if self.patch_learning:
            self.patch_acc = list()
            self.patch_loss = list()
            self.patch_classifier = FlatPatchClassifier(dims=[self.dim for _ in range(4)], num_classes=self.num_classes, cls_type=cls_type)
    
    
    def forward(self, x, mask=None):
        features = self.vt.forward_features(x)
        # Remove cls token
        # Features must be in shape b, c, h, w, but currently in b, n, c
        for fdx, f in enumerate(features):
            feature = f[:, 1:, :]
            feature = feature.permute(0, 2, 1)
            b, c, hw = feature.shape
            
            h = w = int(math.sqrt(hw))
            
            feature = feature.reshape(b, c, h, w)
            features[fdx] = feature

        if self.patch_learning:
            features, loss = self.patch_forward(features, mask)
            x = self.dec(features)
            return x, loss            
        
        return self.dec(features)
    
    def patch_accuracy(self, patch_pred, mask):
        patch_pred = torch.argmax(patch_pred, dim=1)
        patch_acc = acc(patch_pred, mask)
        return patch_acc
    
    
    def patch_forward(self, features, mask):

        patch_loss = torch.zeros(1)

        feature, prediction = self.patch_classifier(features)
            
        if mask is not None:
            feature_mask = patchify_mask(mask, self.patch_size)
            feature_mask = compute_class_proportions(feature_mask, self.num_classes)
            feature_mask = feature_mask.permute(0, 3, 1, 2)
            patch_loss = F.cross_entropy(prediction, feature_mask)
        
        
        self.patch_loss.append(patch_loss.item())            
        
        return feature, patch_loss.mean()

    
    def calculate_metrics(self, logits, mask, step_type="train"):
        prediction = F.softmax(logits, dim=1).argmax(dim=1)

        miou = iou(prediction, mask, self.num_classes)
        macc = acc(prediction, mask, self.num_classes)

        if step_type == "train":
            self.train_iou.append(miou.item())
            self.train_acc.append(macc.item())
        else:
            self.val_iou.append(miou.mean().item())
            self.val_acc.append(macc.mean().item())
    
    def on_train_epoch_end(self):
        if len(self.train_iou) > 0:
            epoch_iou = mean(self.train_iou)
        else:
            epoch_iou = 0
        if len(self.train_acc) > 0:
            epoch_acc = mean(self.train_acc)
        else:
            epoch_acc = 0
        if len(self.train_loss) > 0:
            epoch_loss = mean(self.train_loss)
        else:
            epoch_loss = 0

        self.log("train_loss", epoch_loss, on_epoch=True, sync_dist=True)
        self.log("train_iou", epoch_iou, on_epoch=True, sync_dist=True)
        self.log("train_acc", epoch_acc, on_epoch=True, sync_dist=True)

        if self.patch_learning:
            if len(self.patch_loss) > 0:
                patch_loss = mean(self.patch_loss)
                self.log("train_patch_loss", patch_loss, on_epoch=True, sync_dist=True)
            print(f"Training stats ({self.current_epoch}) | Loss: {epoch_loss}, IoU: {epoch_iou}, Acc: {epoch_acc} Patch Loss: {patch_loss} \n")
        else:
            print(f"Training stats ({self.current_epoch}) | Loss: {epoch_loss}, IoU: {epoch_iou}, Acc: {epoch_acc} \n")
    
    
    def on_validation_epoch_end(self):
        if len(self.val_iou) > 0:
            epoch_iou = mean(self.val_iou)
        else:
            epoch_iou = 0
        if len(self.val_acc) > 0:
            epoch_acc = mean(self.val_acc)
        else:
            epoch_acc = 0
        if len(self.val_loss) > 0:
            epoch_loss = mean(self.val_loss)
        else:
            epoch_loss = 0
        
        self.log("val_loss", epoch_loss, on_epoch=True, sync_dist=True)
        self.log("val_iou", epoch_iou, on_epoch=True, sync_dist=True)
        self.log("val_acc", epoch_acc, on_epoch=True, sync_dist=True)
        
        print(f"Validation stats ({self.current_epoch}) | Loss: {epoch_loss}, IoU: {epoch_iou}, Acc: {epoch_acc} \n")

    def training_step(self, batch, batch_idx):
        image, mask = batch
        
        loss = torch.zeros(1).to(self.device)
        
        if self.patch_learning:
            x, loss = self(image, mask)
        else:
            x = self(image)
        
        loss = loss + self.loss(x, mask)
        
        self.train_loss.append(loss.item())
        
        self.calculate_metrics(x, mask, step_type="train")
        
        return loss


    def validation_step(self, batch, batch_idx):
        image, mask = batch
        
        if self.patch_learning:
            x, loss = self(image)
        else:
            x = self(image)
        
        loss = self.loss(x, mask)
        self.val_loss.append(loss.item())
        
        self.calculate_metrics(x, mask, step_type="val")
        

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
