
from models.dcswin import dcswin_tiny, dcswin_small, dcswin_base
from models.patch_classifier import PatchClassifier

def get_model(model_name, model_size, num_classes):
    if model_name == "dcswin":
        if model_size == "tiny":
            return dcswin_tiny(num_classes=num_classes)
        elif model_size == "small":
            return dcswin_small(num_classes=num_classes)
        elif model_size == "base":
            return dcswin_base(num_classes=num_classes)
        else:
            raise NotImplementedError

def get_patch_classifier(encoder_channels, num_classes, binary):
    return PatchClassifier(encoder_channels, num_classes, binary=binary)
