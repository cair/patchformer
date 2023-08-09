import lightning as L
import torch

import torch.nn.functional as F

from utils.model_utils import get_patch_classifier

from utils.metric_utils import iou_pytorch as iou, acc_pytorch as acc

from statistics import mean


class ViTLightning(L.LightningModule):

    def __init__(self, model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 num_classes: int,
                 learning_rate: float = 1e-4,
                 patch_learning: bool = False,
                 dual: bool = False) -> None:

        super().__init__()

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes

        # self.dice_loss = DiceLoss(mode="multiclass", ignore_index=train_loader.dataset.num_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=self.num_classes, label_smoothing=0.1)

        self.learning_rate = learning_rate
        self.patch_learning = patch_learning

        self.patch_sizes = [4, 8, 16, 32]

        if patch_learning:
            self.model.patch_classifier = get_patch_classifier(self.model.encs, self.num_classes + 1, dual=dual)

        # Training metrics
        self.train_iou = list()
        self.train_acc = list()
        self.train_loss = list()

        # Validation metrics
        self.val_iou = list()
        self.val_acc = list()
        self.val_loss = list()

    def patch_accuracy(self, patch_pred, mask):
        patch_pred = torch.argmax(patch_pred, dim=1)
        patch_acc = acc(patch_pred, mask)
        return patch_acc

    def forward(self, x, mask=None):

        if self.patch_learning:
            x, patch_loss = self.model.patch_forward(x, mask)
        else:
            x, patch_loss = self.model(x, mask)
        """
        if self.patch_learning:
            stems, patch_loss = self.patch_loss(xs, mask)

            for i in range(len(xs)):
                xs[i] = xs[i] + stems[i]

        if mask is not None:
            return patch_loss, self.model.decoder(*xs)
        """
        return patch_loss, x

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
            _, x = self.forward(img)

        # Segmentation loss
        # dice_loss = self.dice_loss(x, mask)
        ce_loss = self.ce_loss(x, mask)

        if self.patch_learning:
            loss_list = torch.cat([ce_loss.unsqueeze(0), patch_losses.mean().unsqueeze(0)], dim=0)
            # loss_list = torch.cat([dice_loss.unsqueeze(0), ce_loss.unsqueeze(0)], dim=0)
        else:
            loss_list = torch.cat([ce_loss.unsqueeze(0)], dim=0)

        loss = torch.sum(loss_list)

        self.log("train_segmentation_loss", ce_loss.cpu().item(), batch_size=img.size(0), on_epoch=True, sync_dist=True)

        self.train_loss.append(loss.item())
        self.calculate_metrics(x, mask, step_type="train")

        self.log("train_loss", loss.cpu().item(), batch_size=img.size(0), on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch["img"], batch["mask"]
        mask = mask.squeeze(1)

        if self.patch_learning:
            # patch_losses, x = self(img, mask)
            _, x = self(img, mask)
        else:
            _, x = self(img)


        # Segmentation loss

        # dice_loss = self.dice_loss(x, mask)
        ce_loss = self.ce_loss(x, mask)

        self.log("val_segmentation_loss", ce_loss.cpu().item(), batch_size=img.size(0), on_epoch=True,
                 sync_dist=True)

        self.val_loss.append(ce_loss.item())
        self.calculate_metrics(x, mask, step_type="val")

        self.log("val_loss", ce_loss.cpu().item(), batch_size=img.size(0), on_epoch=True, sync_dist=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6)
        return [optimizer] , [scheduler]
