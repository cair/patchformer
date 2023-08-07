
from models.dcswin import dcswin_tiny, dcswin_small, dcswin_base
from models.hiera import hiera_tiny_224, hiera_small_224, hiera_base_224
from models.patch_classifier import PatchClassifier, DualPatchClassifier

def get_model(model_name, model_size, num_classes, binary=False, input_size=(256, 256)):
    if model_name == "dcswin":
        if model_size == "tiny":
            return dcswin_tiny(num_classes=num_classes, binary=binary)
        elif model_size == "small":
            return dcswin_small(num_classes=num_classes, binary=binary)
        elif model_size == "base":
            return dcswin_base(num_classes=num_classes, binary=binary)
        else:
            raise NotImplementedError
    elif model_name == "hiera":
        if model_size == "tiny":
            return hiera_tiny_224(num_classes=num_classes, input_size=input_size)
        elif model_size == "small":
            return hiera_small_224(num_classes=num_classes, input_size=input_size)
        elif model_size == "base":
            return hiera_base_224(num_classes=num_classes, input_size=input_size)
        else:
            raise NotImplementedError

def get_patch_classifier(encoder_channels, num_classes, dual=False):

    if dual:
        return DualPatchClassifier(encoder_channels, num_classes)

    return PatchClassifier(encoder_channels, num_classes)
