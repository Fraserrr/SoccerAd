import torch
import warnings
import torch.nn.functional as F
import torch.nn as nn

from functools import partial
from collections import OrderedDict
from torchinfo import summary
from transformers import AutoModel, AutoImageProcessor

##BACKBONE DINOv3 

BACKBONE_SIZE = "small" # in ("small", "base", "large")

# DINOv3 model mapping
backbone_archs = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m", 
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

backbone_model_name = backbone_archs[BACKBONE_SIZE]

# Load DINOv3 model from Hugging Face
backbone_model = AutoModel.from_pretrained(backbone_model_name)

# Move to GPU if available, otherwise keep on CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone_model = backbone_model.to(device)

# Freeze backbone parameters
for name, param in backbone_model.named_parameters():
    param.requires_grad = False

# Get model configuration for dimensions
config = backbone_model.config
patch_size = config.patch_size
hidden_size = config.hidden_size
num_register_tokens = config.num_register_tokens

def resize(input_data,
       size=None,
       scale_factor=None,
       mode='nearest',
       align_corners=None,
       warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input_data, size, scale_factor, mode, align_corners)

class BNHead(nn.Module):
    """Head for DINOv3 features with batch normalization."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        # Use the hidden size from DINOv3 configuration
        self.in_channels = hidden_size
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors
        self.in_index = [0]  # We'll use only the patch features
        self.input_transform = 'resize_concat'
        self.align_corners = False

        self.conv_seg = nn.Conv2d(self.in_channels, 21, kernel_size=1)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (Tensor): DINOv3 last_hidden_state tensor.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.bn(x)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs from DINOv3 output format to spatial format.
        Args:
            inputs (Tensor): DINOv3 last_hidden_state of shape 
                            [batch_size, 1 + num_register_tokens + num_patches, hidden_size]
        Returns:
            Tensor: The transformed inputs in spatial format [batch_size, hidden_size, H, W]
        """
        # inputs shape: [batch_size, 1 + num_register_tokens + num_patches, hidden_size]
        batch_size, seq_len, hidden_dim = inputs.shape
        
        # Skip CLS token and register tokens, keep only patch tokens
        patch_tokens = inputs[:, 1 + num_register_tokens:, :]  # [batch_size, num_patches, hidden_size]
        
        # Calculate spatial dimensions from patch tokens count
        num_patches = patch_tokens.shape[1]
        
        # Try to determine spatial dimensions based on input
        # For DINOv3 with patch_size=16, if input is HxW, patches should be (H//16) x (W//16)
        patch_h = patch_w = int(num_patches ** 0.5)
        
        # Check if it's a perfect square
        if patch_h * patch_w != num_patches:
            # Not a perfect square - try to find the best rectangle dimensions
            # This happens when input image is not square
            
            # Find factors of num_patches
            factors = []
            for i in range(1, int(num_patches**0.5) + 1):
                if num_patches % i == 0:
                    factors.append((i, num_patches // i))
            
            if factors:
                # Choose the factor pair that's closest to square
                patch_h, patch_w = min(factors, key=lambda x: abs(x[0] - x[1]))
                print(f"🔧 Using patch dimensions: {patch_h}x{patch_w} for {num_patches} patches")
            else:
                # Fallback: use square and truncate extra patches
                patch_h = patch_w = int(num_patches ** 0.5)
                patch_tokens = patch_tokens[:, :patch_h * patch_w, :]
                print(f"⚠️ Truncating patches from {num_patches} to {patch_h * patch_w}")
        
        # Reshape to spatial format: [batch_size, hidden_size, H, W]
        patch_features = patch_tokens.transpose(1, 2).reshape(batch_size, hidden_dim, patch_h, patch_w)
        
        return patch_features

    def cls_seg(self, feat):
        """Classify each pixel."""
        output = self.conv_seg(feat)
        return output
        
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    
class DINOv3Segmentation(nn.Module):
    def __init__(self):
        super(DINOv3Segmentation, self).__init__()

        self.backbone_model = backbone_model
        self.decode_head = BNHead()

    def forward(self, x):
        # Get DINOv3 features
        outputs = self.backbone_model(pixel_values=x)
        # outputs.last_hidden_state shape: [batch_size, 1 + num_register_tokens + num_patches, hidden_size]
        
        # Pass to decode head
        segmentation_output = self.decode_head(outputs.last_hidden_state)
        
        return segmentation_output
    
if __name__ == '__main__':
    model = DINOv3Segmentation()
    summary(
        model, 
        (1, 3, 224, 224),  # DINOv3 standard input size
        col_names=('input_size', 'output_size', 'num_params'),
        row_settings=['var_names']
    )