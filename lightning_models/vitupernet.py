import lightning as L
import torch
import torch.nn.functional as F

from utils import iou_pytorch as iou, acc_pytorch as acc

import math
from statistics import mean

class ViTUperNet(L.LightningModule):
    def __init__(self,
                 vision_transformer: torch.nn.Module,
                 decoder: torch.nn.Module,
                 num_classes: int,
                 learning_rate: float,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 patch_learning: bool = False,
                 ):
        super().__init__()
        
        self.vt = vision_transformer
        self.dec = decoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.num_classes, label_smoothing=0.1)
        
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
    
    def forward(self, x):
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
            
        
        x = self.dec(features)
        
        return x
    
    def calculate_metrics(self, logits, mask, step_type="train"):
        prediction = F.softmax(logits, dim=1).argmax(dim=1)

        miou = iou(prediction, mask)
        macc = acc(prediction, mask)

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

        #self.log("train_loss", epoch_loss, on_epoch=True, sync_dist=True)
        #self.log("train_iou", epoch_iou, on_epoch=True, sync_dist=True)
        #self.log("train_acc", epoch_acc, on_epoch=True, sync_dist=True)

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
        
        print(f"Validation stats ({self.current_epoch}) | Loss: {epoch_loss}, IoU: {epoch_iou}, Acc: {epoch_acc} \n")

    def training_step(self, batch, batch_idx):
        image, mask = batch
        
        x = self(image)

        loss = self.loss(x, mask)
        
        self.train_loss.append(loss.item())
        
        self.calculate_metrics(x, mask, step_type="train")
        
        return loss


    def validation_step(self, batch, batch_idx):
        image, mask = batch
        
        x = self(image) # Shape (B, 785, 768)
        
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