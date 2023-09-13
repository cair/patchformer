import torch
import lightning as L
from timm import create_model

from loaders.loader import get_dataloader
from models.decoders.upernet import UPerNet
from lightning_models.vitupernet import ViTUperNet

def main():
    
    transformer_model = create_model("vit_base_patch16_224", pretrained=True)
    upernet_decoder = UPerNet(num_class=182, fc_dim=768, fpn_inplanes=(768, 768, 768, 768), fpn_dim=512)
    
    train_loader = get_dataloader(dataset_name="coco", dataset_type="train", batch_size=8, percentage=0.01)
    val_loader = get_dataloader(dataset_name="coco", dataset_type="val", batch_size=1, percentage=0.01)
    
    lightning_model = ViTUperNet(vision_transformer=transformer_model,
                                 decoder=upernet_decoder,
                                 num_classes=182,
                                 learning_rate=1e-4,
                                 train_loader=train_loader,
                                 val_loader=val_loader)
    
    trainer = L.Trainer(max_epochs=5, accelerator="gpu", devices=[0])
    trainer.fit(model=lightning_model)
    
    exit("here")
    
    train_loader = get_dataloader(dataset_name="coco", dataset_type="train", batch_size=8, percentage=0.01)
    
    for batch in train_loader:
        image, mask = batch
        print(image.shape)
        print(mask.shape)
        exit("")


if __name__ == "__main__":
    main()