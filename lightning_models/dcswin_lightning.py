import lightning as L
import torch

import torch.nn.functional as F

from utils.metric_utils import iou_pytorch as iou, acc_pytorch as acc
from utils.patch_utils import patchify_mask, check_homogeneity_binary, check_homogeneity_classes
from losses.dice import DiceLoss

from statistics import mean


class DCSwinLightning(L.LightningModule):

    def __init__(self, model: torch.nn.Module,
                 patch_classifiers: torch.nn.ModuleList,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 learning_rate: float = 1e-4,
                 patch_learning: bool = False,
                 binary: bool = False) -> None:

        super().__init__()

        self.model = model
        if patch_classifiers is not None:
            self.patch_classifiers = patch_classifiers.patch_classifiers
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.dice_loss = DiceLoss(mode="multiclass", ignore_index=train_loader.dataset.num_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=train_loader.dataset.num_classes)

        self.learning_rate = learning_rate
        self.patch_learning = patch_learning
        self.binary = binary

        self.patch_sizes = [4, 8, 16, 32]

        # Training metrics
        self.train_iou = list()
        self.train_acc = list()
        self.train_loss = list()

        # Validation metrics
        self.val_iou = list()
        self.val_acc = list()
        self.val_loss = list()

    def binary_patch_loss(self, x, mask):

        patch_loss = torch.zeros((4,), device=self.device)

        for xidx, xi in enumerate(x):
            pc = self.patch_classifiers[xidx]
            ps = self.patch_sizes[xidx]
            maski = patchify_mask(mask, ps)
            maski = check_homogeneity_binary(maski, ignore_index=2)
            assert xi.size(2) == maski.size(1)  # xi (b, c, h, w), maski (b, h, w, p)
            assert xi.size(3) == maski.size(2)
            xi = F.sigmoid(pc(xi).permute(0, 3, 1, 2))
            inverse_xi = 1 - xi
            xi = torch.cat((inverse_xi, xi), dim=1)
            l = self.ce_loss(xi, maski)
            patch_loss[xidx] = l

        return patch_loss

    def multiclass_patch_loss(self, x, mask):
        pass

    def patch_loss(self, x, mask):
        if self.binary:
            return self.binary_patch_loss(x, mask)
        else:
            return self.multiclass_patch_loss(x, mask)


    def forward(self, x, mask=None):

        (x1, x2, x3, x4), x = self.model(x)

        if mask is not None:
            patch_loss = self.patch_loss([x1, x2, x3, x4], mask)
            return patch_loss, x

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

        self.log("train_loss", epoch_loss, on_epoch=True, sync_dist=True)
        self.log("train_iou", epoch_iou, on_epoch=True, sync_dist=True)
        self.log("train_acc", epoch_acc, on_epoch=True, sync_dist=True)

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
        img, mask = batch["img"], batch["mask"]
        mask = mask.squeeze(1)

        if self.patch_learning:
            patch_losses, x = self.forward(img, mask)
        else:
            x = self.forward(img)

        # Segmentation loss
        dice_loss = self.dice_loss(x, mask)
        ce_loss = self.ce_loss(x, mask)

        if self.patch_learning:
            loss_list = torch.cat([dice_loss.unsqueeze(0), ce_loss.unsqueeze(0), patch_losses.mean().unsqueeze(0)], dim=0)
        else:
            loss_list = torch.cat([dice_loss.unsqueeze(0), ce_loss.unsqueeze(0)], dim=0)

        loss = torch.sum(loss_list)

        self.log("train_segmentation_loss", dice_loss.cpu().item() + ce_loss.cpu().item(), batch_size=img.size(0), on_epoch=True, sync_dist=True)

        self.train_loss.append(loss.item())
        self.calculate_metrics(x, mask, step_type="train")

        self.log("train_loss", loss.cpu().item(), batch_size=img.size(0), on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch["img"], batch["mask"]
        mask = mask.squeeze(1)

        if self.patch_learning:
            patch_losses, x = self(img, mask)
        else:
            x = self(img)


        # Segmentation loss

        dice_loss = self.dice_loss(x, mask)
        ce_loss = self.ce_loss(x, mask)

        if self.patch_learning:
            loss_list = torch.cat([dice_loss.unsqueeze(0), ce_loss.unsqueeze(0), patch_losses.mean().unsqueeze(0)], dim=0)
        else:
            loss_list = torch.cat([dice_loss.unsqueeze(0), ce_loss.unsqueeze(0)], dim=0)

        loss = torch.sum(loss_list)

        self.log("val_segmentation_loss", dice_loss.cpu().item() + ce_loss.cpu().item(), batch_size=img.size(0), on_epoch=True,
                 sync_dist=True)

        self.val_loss.append(loss.item())
        self.calculate_metrics(x, mask, step_type="val")

        self.log("val_loss", loss.cpu().item(), batch_size=img.size(0), on_epoch=True, sync_dist=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=1e-6)
        return [optimizer]  # , [scheduler]
