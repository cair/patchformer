from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

import cv2
import torch
import numpy as np

from PIL import Image

import torch.nn.functional as F

from utils.metric_utils import iou_pytorch as iou

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from utils.model_utils import get_model
from lightning_models.dcswin_lightning import DCSwinLightning
from loaders.dataset import get_datasets
from loaders.loader import get_loader


def segmentation_to_image(image):
    """
    Function that takes in a prediction as a tensor and converts it to an image,
    an saves it as a grayscale png file.
    :param image:
    :return:
    """
    print(image.shape)
    image = torch.argmax(torch.softmax(image, dim=1), dim=1)
    image = image.squeeze(0).unsqueeze(-1).detach().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    cv2.imwrite("test.png", image)
    print("Wrote predicted mask to 'test.png'")


def reshape_transform(tensor, height=8, width=8):
    if tensor.shape[1] == 8 * 8:
        result = tensor.reshape(tensor.size(0), 8, 8, tensor.size(2))
    elif tensor.shape[1] == 16 * 16:
        result = tensor.reshape(tensor.size(0), 16, 16, tensor.size(2))
    elif tensor.shape[1] == 32 * 32:
        result = tensor.reshape(tensor.size(0), 32, 32, tensor.size(2))
    elif tensor.shape[1] == 64 * 64:
        result = tensor.reshape(tensor.size(0), 64, 64, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class PlaceHolder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, x = self.model(x)
        return x


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.mask = self.mask.cuda()
        print("mask:", self.mask.shape)

    def __call__(self, model_output):
        print("model_output:", model_output.shape)
        return (model_output[self.category, :, :] * self.mask).sum()


if __name__ == "__main__":

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    model, model_size, num_classes,  = "dcswin", "tiny", 2

    model = get_model(model, model_size, num_classes)

    train_dataset, val_dataset = get_datasets("inria", 256)
    train_loader, val_loader = get_loader(train_dataset, 8, True, 4), get_loader(val_dataset, 1, False, 4)

    for batch in train_loader:
        image, mask, img_id = batch["img"], batch["mask"], batch["img_id"]
        break

    num_classes = train_dataset.num_classes

    lm = DCSwinLightning.load_from_checkpoint("lightning_logs/dcswin_inria_256_tiny/epoch=16-val_loss=0.37.ckpt", "cpu", model=model, train_loader=train_loader, val_loader=val_loader, num_classes=num_classes, learning_rate=0.0001,
                         patch_learning=True, binary=False)


    #image = image[0].unsqueeze(0)
    #mask = mask[0].unsqueeze(0)
    img_id = img_id[0]

    print(image.min(), image.max())

    print(image.shape)
    print(mask.shape)

    #mask = torch.tensor(mask).unsqueeze(0) / 255.0
    #mask = mask.to(torch.long)
    #print(mask.min(), mask.max())


    model = PlaceHolder(lm)

    target_layer_0 = model.model.model.backbone.layers[0].blocks[-1].norm1
    target_layer_1 = model.model.model.backbone.layers[1].blocks[-1].norm1
    target_layer_2 = model.model.model.backbone.layers[2].blocks[-1].norm1
    target_layer_3 = model.model.model.backbone.layers[3].blocks[-1].norm1

    output = model(image)

    normalized_mask = torch.nn.functional.softmax(output, dim=1).cpu()

    car_category = 1

    building_mask = normalized_mask[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    building_mask_float = np.float32(building_mask == car_category)

    output_image = Image.fromarray(np.uint8(building_mask * 255))
    output_image.save("output.png")

    target_layers = [target_layer_0, target_layer_1, target_layer_2, target_layer_3]
    targets = [SemanticSegmentationTarget(car_category, building_mask_float)]

    rgb_image = Image.open("data/inria/train/images/" + img_id + ".png")
    mask = Image.open("data/inria/train/masks/" + img_id + ".png")

    rgb_image = rgb_image.resize((256, 256))
    mask = mask.resize((256, 256))

    rgb_image.save("image.png")
    mask.save("mask.png")

    model.cuda()
    image.cuda()

    image = image[0].unsqueeze(0)

    print(image.shape)

    with GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_transform, use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=image, targets=targets)[0, :]
        print(rgb_image)
        rgb_image = np.asarray(rgb_image)
        rgb_image = rgb_image / 255.0
        cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    Image.fromarray(cam_image)

    cv2.imwrite("cam.png", cam_image)
