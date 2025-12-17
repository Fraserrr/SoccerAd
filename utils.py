import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
import albumentations as A
import torch.nn as nn

from config import (
    VIS_LABEL_MAP as viz_map
)

plt.style.use('ggplot')

def safe_torch_load(file_path, map_location=None):
    """
    Safely load a PyTorch checkpoint file with fallback for compatibility.
    
    Args:
        file_path: Path to the checkpoint file
        map_location: Device to map the tensors to
    
    Returns:
        Loaded checkpoint dictionary
    """
    try:
        # Try with weights_only=True first (secure)
        return torch.load(file_path, map_location=map_location, weights_only=True)
    except Exception as e:
        print(f"⚠️ Warning: Loading with weights_only=False (trusted source assumed)")
        print(f"   Reason: {e}")
        # Fallback to weights_only=False for older checkpoints
        return torch.load(file_path, map_location=map_location, weights_only=False)

def calculate_optimal_dimensions(width, height, patch_size=16, padding=4):
    """
    Calculate optimal image dimensions that work with DINOv3 patch size and dataset padding.
    
    Args:
        width: Original image width
        height: Original image height 
        patch_size: DINOv3 patch size (default: 16)
        padding: Dataset padding (default: 4)
        
    Returns:
        tuple: (optimal_width, optimal_height)
    """
    # After padding, dimensions must be divisible by patch_size
    optimal_width = ((width + padding) // patch_size) * patch_size - padding
    optimal_height = ((height + padding) // patch_size) * patch_size - padding
    
    # Ensure dimensions are positive
    optimal_width = max(optimal_width, patch_size - padding)
    optimal_height = max(optimal_height, patch_size - padding)
    
    return optimal_width, optimal_height

def validate_dinov3_dimensions(width, height, patch_size=16, padding=4):
    """
    Validate if image dimensions are compatible with DINOv3.
    
    Returns:
        bool: True if dimensions are valid
    """
    return ((width + padding) % patch_size == 0) and ((height + padding) % patch_size == 0)

def calculate_dinov3_dimensions(target_width, target_height, patch_size=16, padding=4):
    """
    Calculate optimal dimensions for DINOv3 input considering padding.
    
    DINOv3 requirements:
    - Dataset adds 4 pixels of padding to each dimension
    - Final padded size must be divisible by patch_size (16)
    - Formula: (input_size + padding) % patch_size == 0
    
    Args:
        target_width: Desired width
        target_height: Desired height  
        patch_size: DINOv3 patch size (default: 16)
        padding: Padding added by dataset (default: 4)
    
    Returns:
        tuple: (optimal_width, optimal_height)
    """
    def find_optimal_size(target_size):
        # We need (size + padding) % patch_size == 0
        # So size % patch_size == (patch_size - padding) % patch_size
        remainder_needed = (patch_size - padding) % patch_size
        
        # Find closest size that satisfies the condition
        if target_size % patch_size == remainder_needed:
            return target_size
        
        # Try smaller and larger sizes
        smaller = target_size - (target_size % patch_size) + remainder_needed
        if smaller <= 0:
            smaller += patch_size
            
        larger = smaller + patch_size
        
        # Choose the closest to target
        if abs(target_size - smaller) <= abs(target_size - larger):
            return smaller
        else:
            return larger
    
    optimal_width = find_optimal_size(target_width)
    optimal_height = find_optimal_size(target_height)
    
    return optimal_width, optimal_height

def print_dinov3_dimension_info(width, height, patch_size=16, padding=4):
    """
    Print information about DINOv3 dimension compatibility.
    
    Args:
        width: Input width
        height: Input height
        patch_size: DINOv3 patch size (default: 16)
        padding: Padding added by dataset (default: 4)
    """
    padded_width = width + padding
    padded_height = height + padding
    
    print(f"📐 DINOv3 Dimension Analysis:")
    print(f"   Input size: {width}x{height}")
    print(f"   After padding (+{padding}): {padded_width}x{padded_height}")
    print(f"   Patch size: {patch_size}x{patch_size}")
    
    width_patches = padded_width // patch_size
    height_patches = padded_height // patch_size
    width_remainder = padded_width % patch_size
    height_remainder = padded_height % patch_size
    
    print(f"   Patches: {width_patches}x{height_patches}")
    
    if width_remainder == 0 and height_remainder == 0:
        print(f"   ✅ Perfect fit! Total patches: {width_patches * height_patches}")
    else:
        print(f"   ⚠️ Remainder: {width_remainder}x{height_remainder}")
        optimal_w, optimal_h = calculate_dinov3_dimensions(width, height, patch_size, padding)
        print(f"   💡 Suggested dimensions: {optimal_w}x{optimal_h}")
        
        # Show what the suggested dimensions would give
        new_padded_w = optimal_w + padding
        new_padded_h = optimal_h + padding
        new_patches_w = new_padded_w // patch_size  
        new_patches_h = new_padded_h // patch_size
        print(f"   → After padding: {new_padded_w}x{new_padded_h}")
        print(f"   → Patches: {new_patches_w}x{new_patches_h} = {new_patches_w * new_patches_h} total")

def set_class_values(all_classes, classes_to_train):
    """
    This (`class_values`) assigns a specific class label to the each of the classes.
    For example, `animal=0`, `archway=1`, and so on.

    :param all_classes: List containing all class names.
    :param classes_to_train: List containing class names to train.
    """
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label

    :param mask: NumPy array, segmentation mask.
    :param class_values: List containing class values, e.g car=0, bus=1.
    :param label_colors_list: List containing RGB color value for each class.
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
    label_mask = label_mask.astype(int)
    return label_mask

def denormalize(x, mean=None, std=None):
    # x should be a Numpy array of shape [H, W, C] 
    x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    res = torch.clamp(t, 0, 1)
    res = res.squeeze(0).permute(1, 2, 0).numpy()
    return res

def draw_translucent_seg_maps(
    data, 
    output, 
    epoch, 
    i, 
    val_seg_dir, 
    label_colors_list,
):
    """
    This function color codes the segmentation maps that is generated while
    validating. THIS IS NOT TO BE CALLED FOR SINGLE IMAGE TESTING
    """
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    alpha = 1 # how much transparency
    beta = 0.8 # alpha + beta should be 1
    gamma = 0 # contrast

    seg_map = output[0] # use only one output from the batch
    seg_map = torch.argmax(seg_map.squeeze(), dim=0).detach().cpu().numpy()

    image = denormalize(data[0].permute(1, 2, 0).cpu().numpy(), IMG_MEAN, IMG_STD)

    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)

    for label_num in range(0, len(label_colors_list)):
        index = seg_map == label_num
        red_map[index] = np.array(viz_map)[label_num, 0]
        green_map[index] = np.array(viz_map)[label_num, 1]
        blue_map[index] = np.array(viz_map)[label_num, 2]
        
    rgb = np.stack([red_map, green_map, blue_map], axis=2)
    rgb = np.array(rgb, dtype=np.float32)
    # convert color to BGR format for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float32) * 255.
    # cv2.imshow('rgb', rgb)
    # cv2.waitKey(0)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
  
    # Save the colored overlay
    cv2.addWeighted(image, alpha, rgb, beta, gamma, image)
    cv2.imwrite(f"{val_seg_dir}/e{epoch}_b{i}.jpg", image)

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, epoch, model, out_dir, name='model'
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))

class SaveBestModelIOU:
    """
    Class to save the best model while training. If the current epoch's 
    IoU is higher than the previous highest, then save the
    model state.
    """
    def __init__(self, best_iou=float(0)):
        self.best_iou = best_iou
        
    def __call__(self, current_iou, epoch, model, out_dir, name='model'):
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            print(f"\nBest validation IoU: {self.best_iou}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))

def save_model(epochs, model, optimizer, criterion, out_dir, name='model'):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(out_dir, name+'.pth'))

def save_plots(
    train_acc, valid_acc, 
    train_loss, valid_loss, 
    train_miou, valid_miou, 
    out_dir
):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))

    # mIOU plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_miou, color='tab:blue', linestyle='-', 
        label='train mIoU'
    )
    plt.plot(
        valid_miou, color='tab:red', linestyle='-', 
        label='validataion mIoU'
    )
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'miou.png'))

# Define the torchvision image transforms
def infer_transform(img_size):
    transform = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
        A.PadIfNeeded(
                min_height=img_size[1]+4, 
                min_width=img_size[0]+4,
                position='center',
                value=0,
                mask_value=0
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.
        )
    ])
    return transform

def get_segment_labels(image, model, device, img_size):
    # transform the image to tensor and load into computation device
    infer_tfms = infer_transform(img_size)
    transformed_image = infer_tfms(image=image)['image']
    transformed_image = transformed_image.transpose(2, 0, 1)
    image_tensor = torch.tensor(transformed_image).unsqueeze(0).to(device) # add a batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)

    upsampled_logits = nn.functional.interpolate(
                outputs, size=image.shape[0:2], 
                mode="bilinear", 
                align_corners=False
            ).squeeze(0)
    
    return upsampled_logits

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs, dim=0).detach().cpu().numpy()

    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(viz_map)):
        index = labels == label_num
        red_map[index] = np.array(viz_map)[label_num, 0]
        green_map[index] = np.array(viz_map)[label_num, 1]
        blue_map[index] = np.array(viz_map)[label_num, 2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map


def image_overlay(image, segmented_image):
    """
    :param image: Image in RGB format.
    :param segmented_image: Segmentation map in RGB format. 
    """
    alpha = 0.8 # transparency for the original image
    beta = 1.0 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image