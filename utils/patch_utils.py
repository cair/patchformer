import torch

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

def check_homogeneity_binary(tensor, ignore_index):
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
    tensor_bool = tensor_bool.all(dim=-1)

    # Reshape the tensor back to original shape (minus last dimension)
    tensor_bool = tensor_bool.reshape(original_shape[:-1])

    return tensor_bool.long()

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