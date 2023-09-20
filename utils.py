import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from datetime import datetime


SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch


def acc_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze(1)
    acc = torch.sum(outputs == labels) / (labels.size(0) * labels.size(1) * labels.size(2))

    return acc


def patchify_mask(mask, patch_size):
    # Input mask is of size (b, h, w)
    # Get batch size, height and width
    b, h, w = mask.shape

    # Create patches using unfold. The resulting size is (b, h', w', patch_size, patch_size)
    # where h' and w' are the number of patches in height and width dimension respectively
    patches = mask.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

    # Now we have to change the shape to (b, h', w', patch_size*patch_size)
    patched_mask = patches.contiguous().view(b, h // patch_size, w // patch_size, patch_size * patch_size)

    return patched_mask


def check_homogeneity_classes(tensor, num_classes, ignore_index):
    # Get the shape of the tensor
    original_shape = tensor.shape
    last_dim = original_shape[-1]

    # Reshape the tensor to 2D for easy comparison
    tensor = tensor.reshape(-1, last_dim)

    # Create a mask for elements to ignore
    ignore_mask = tensor.eq(ignore_index)

    # Compare each element with the first along the last dimension, ignoring specified class
    tensor_bool = tensor.eq(tensor[:, 0].unsqueeze(1)) & ~ignore_mask

    # Check if all elements along the last dimension are the same, ignoring specified class
    tensor_homogeneous = tensor_bool.all(dim=-1)

    # Get the value of the first element in each row, excluding ignored class
    first_elements = tensor[:, 0]
    first_elements = torch.where(ignore_mask[:, 0], torch.tensor(num_classes), first_elements)

    # Create a tensor where each position indicates the class of the homogenous slice,
    # or num_classes if the slice is heterogenous or ignored.
    class_tensor = torch.where(tensor_homogeneous, first_elements, torch.tensor(num_classes))

    # Reshape the class tensor back to the original shape (minus last dimension)
    class_tensor = class_tensor.reshape(original_shape[:-1])

    return class_tensor

def check_homogeneity_majority(tensor, num_classes, ignore_index):

    # Get the shape of the tensor
    original_shape = tensor.shape
    last_dim = original_shape[-1]

    # Reshape the tensor to 2D for easy comparison
    tensor = tensor.reshape(-1, last_dim)

    # Create a mask for elements to ignore
    ignore_mask = tensor.eq(ignore_index)

    # Compare each element with the first along the last dimension, ignoring specified class
    tensor_bool = tensor.eq(tensor[:, 0].unsqueeze(1)) & ~ignore_mask

    # Check if all elements along the last dimension are the same, ignoring specified class
    tensor_homogeneous = tensor_bool.all(dim=-1)

    # Calculate the majority class along the last dimension, ignoring specified class
    majority_class_mask = ~ignore_mask
    majority_classes = torch.where(
        majority_class_mask, 
        tensor, 
        torch.full_like(tensor, fill_value=num_classes)
    )

    majority_class_count = majority_classes.bincount(minlength=num_classes+1)[:-1]
    majority_class = majority_class_count.argmax()

    # Calculate the proportion of the majority class
    majority_class_proportion = majority_class_count[majority_class].float() / majority_class_mask.sum(dim=-1).float()
    
    # Assign the majority class if it constitutes more than 50% of the elements
    class_tensor = torch.where(
        majority_class_proportion > 0.5, 
        majority_class, 
        torch.tensor(num_classes)
    )

    # If the slice was homogeneous, assign the class of the first element
    first_elements = tensor[:, 0]
    first_elements = torch.where(ignore_mask[:, 0], torch.tensor(num_classes), first_elements)
    class_tensor = torch.where(tensor_homogeneous, first_elements, class_tensor)

    # Reshape the class tensor back to the original shape (minus last dimension)
    class_tensor = class_tensor.reshape(original_shape[:-1])

    return class_tensor


def check_homogeneity_proportions(tensor, num_classes, ignore_index):

    # Get the shape of the tensor
    original_shape = tensor.shape
    last_dim = original_shape[-1]

    # Reshape the tensor to 2D for easy comparison
    tensor = tensor.reshape(-1, last_dim)

    # Create a mask for elements to ignore
    ignore_mask = tensor.eq(ignore_index)

    # Get the mask for valid elements (not equal to ignore_index)
    valid_mask = ~ignore_mask

    # Get a tensor representation of the indices (used for gathering counts)
    indices = torch.arange(num_classes).unsqueeze(0).unsqueeze(0).to(tensor.device)

    # Calculate the counts of each class along the last dimension, excluding the ignore_index
    class_counts = (tensor.unsqueeze(-1) == indices).float() * valid_mask.unsqueeze(-1)
    class_counts = class_counts.sum(dim=-2)

    # Calculate the proportions of each class
    valid_counts = valid_mask.float().sum(dim=-1, keepdim=True)
    class_proportions = class_counts / valid_counts

    # Reshape the proportions tensor back to the original shape (minus last dimension) and add a new dimension for classes
    class_proportions = class_proportions.reshape(*original_shape[:-1], num_classes)

    return class_proportions

def get_logger(wandb: bool, project: str, name: str, group: str, model: L.LightningModule):
    
    # Create default logger (CSVLogger)
    logger = CSVLogger("lightning_logs", name)
    loggers = [logger]

    if wandb:
        if group is not None:
            wandblogger = WandbLogger(name=name, project=project, group=group)
        else:
            wandblogger = WandbLogger(name=name, project=project)
        wandblogger.watch(model, log_graph=False)
        # wandblogger.experiment.config.update(config)
        loggers.append(wandblogger)

    return loggers

def get_callbacks(project, name):

    config = {}

    config["save_top_k"] = 1
    config["monitor"] = "val_loss"
    config["mode"] = "max"

    config["dirpath"] = "lightning_logs" + "/" + project
    config["filename"] = name + datetime.now().strftime("%d/%m/%Y %H:%M:%S") \
        .replace("/", "").replace(" ", "_").replace(":", "")

    callbacks = []

    callbacks.append(ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_iou",
        mode="max",
        filename="{epoch:02d}-{val_iou:.2f}",
        dirpath=config["dirpath"]
    ))

    return callbacks
