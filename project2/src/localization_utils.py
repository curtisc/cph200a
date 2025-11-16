"""
Utility functions for localization metrics and visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import wandb


def compute_localization_metrics(attention_map: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Compute localization quality metrics (IoU, Dice) between attention map and ground truth mask.

    Args:
        attention_map: Predicted spatial attention weights (B, 1, D', H', W')
        mask: Ground truth localization masks (B, 1, D, H, W)
        threshold: Threshold to binarize attention map (default: 0.5)

    Returns:
        dict: Dictionary with 'iou', 'dice', and 'num_samples' metrics
    """
    # Resize mask to match attention map spatial dimensions
    if attention_map.shape != mask.shape:
        mask_resized = F.interpolate(mask, size=attention_map.shape[2:], mode='trilinear', align_corners=False)
    else:
        mask_resized = mask

    # Binarize attention map at threshold
    attention_binary = (attention_map > threshold).float()
    mask_binary = (mask_resized > 0.5).float()

    # Only compute for samples that have localization annotations
    has_annotation = mask_binary.sum(dim=(2, 3, 4)) > 0  # (B, 1)

    if has_annotation.sum() == 0:
        return {'iou': 0.0, 'dice': 0.0, 'num_samples': 0}

    # Filter to samples with annotations
    attention_binary = attention_binary[has_annotation.squeeze()]
    mask_binary = mask_binary[has_annotation.squeeze()]

    # Compute intersection and union
    intersection = (attention_binary * mask_binary).sum(dim=(1, 2, 3, 4))
    union = (attention_binary + mask_binary).clamp(0, 1).sum(dim=(1, 2, 3, 4))
    attention_sum = attention_binary.sum(dim=(1, 2, 3, 4))
    mask_sum = mask_binary.sum(dim=(1, 2, 3, 4))

    # IoU (Intersection over Union)
    # IoU = |A ∩ B| / |A ∪ B|
    iou = (intersection / (union + 1e-6)).mean().item()

    # Dice coefficient (F1 score for segmentation)
    # Dice = 2 * |A ∩ B| / (|A| + |B|)
    dice = (2 * intersection / (attention_sum + mask_sum + 1e-6)).mean().item()

    return {
        'iou': iou,
        'dice': dice,
        'num_samples': has_annotation.sum().item()
    }


def create_localization_visualization(
    ct_volume: torch.Tensor,
    attention_map: torch.Tensor,
    mask: torch.Tensor,
    num_slices: int = 3,
    slice_axis: int = 2
) -> wandb.Image:
    """
    Create visualization of CT scan with attention overlay and ground truth mask.

    Args:
        ct_volume: CT scan volume (1, D, H, W)
        attention_map: Predicted spatial attention weights (1, D', H', W')
        mask: Ground truth localization mask (1, D, H, W)
        num_slices: Number of slices to visualize (default: 3)
        slice_axis: Which axis to slice along - 0=D, 1=H, 2=W (default: 2=depth)

    Returns:
        wandb.Image: Visualization image with CT, attention, mask, and overlay
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Convert to numpy and remove batch/channel dims
    # Convert to float32 first to handle BFloat16 from mixed precision training
    ct_np = ct_volume.squeeze().float().cpu().numpy()  # (D, H, W)
    attention_np = attention_map.squeeze().float().cpu().numpy()  # (D', H', W')
    mask_np = mask.squeeze().float().cpu().numpy()  # (D, H, W)

    # Resize attention map to match CT volume
    if attention_np.shape != ct_np.shape:
        attention_torch = attention_map.unsqueeze(0).float()  # (1, 1, D', H', W')
        attention_resized = F.interpolate(
            attention_torch,
            size=ct_np.shape,
            mode='trilinear',
            align_corners=False
        )
        attention_np = attention_resized.squeeze().cpu().numpy()

    # Select slices to visualize
    depth = ct_np.shape[slice_axis]
    slice_indices = np.linspace(depth // 4, 3 * depth // 4, num_slices, dtype=int)

    # Create figure with subplots
    fig, axes = plt.subplots(num_slices, 4, figsize=(16, 4 * num_slices))
    if num_slices == 1:
        axes = axes.reshape(1, -1)

    for i, slice_idx in enumerate(slice_indices):
        # Extract slices
        if slice_axis == 0:  # Axial
            ct_slice = ct_np[slice_idx, :, :]
            attn_slice = attention_np[slice_idx, :, :]
            mask_slice = mask_np[slice_idx, :, :]
        elif slice_axis == 1:  # Coronal
            ct_slice = ct_np[:, slice_idx, :]
            attn_slice = attention_np[:, slice_idx, :]
            mask_slice = mask_np[:, slice_idx, :]
        else:  # Sagittal (slice_axis == 2)
            ct_slice = ct_np[:, :, slice_idx]
            attn_slice = attention_np[:, :, slice_idx]
            mask_slice = mask_np[:, :, slice_idx]

        # 1. CT scan only
        axes[i, 0].imshow(ct_slice, cmap='gray')
        axes[i, 0].set_title(f'CT Slice {slice_idx}')
        axes[i, 0].axis('off')

        # 2. Ground truth mask overlay
        axes[i, 1].imshow(ct_slice, cmap='gray')
        if mask_slice.max() > 0:
            axes[i, 1].imshow(mask_slice, cmap='Reds', alpha=0.5 * (mask_slice > 0))
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')

        # 3. Attention map heatmap
        axes[i, 2].imshow(ct_slice, cmap='gray')
        im = axes[i, 2].imshow(attn_slice, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[i, 2].set_title('Predicted Attention')
        axes[i, 2].axis('off')

        # 4. Overlay comparison (mask in red, attention in green)
        axes[i, 3].imshow(ct_slice, cmap='gray')
        if mask_slice.max() > 0:
            axes[i, 3].imshow(mask_slice, cmap='Reds', alpha=0.4 * (mask_slice > 0))
        axes[i, 3].imshow(attn_slice > 0.5, cmap='Greens', alpha=0.4)
        axes[i, 3].set_title('Overlay (Red=GT, Green=Pred)')
        axes[i, 3].axis('off')

    # Adjust layout first, then add legend and colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])  # Leave space for colorbar and legend

    # Add legend for the overlay
    red_patch = mpatches.Patch(color='red', alpha=0.4, label='Ground Truth')
    green_patch = mpatches.Patch(color='green', alpha=0.4, label='Predicted')
    yellow_patch = mpatches.Patch(color='yellow', alpha=0.4, label='Overlap')
    fig.legend(handles=[red_patch, green_patch, yellow_patch],
              loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98))

    # Add colorbar for attention heatmap
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Attention Weight')

    # Convert to wandb Image
    wandb_img = wandb.Image(fig)
    plt.close(fig)

    return wandb_img


def log_localization_visualizations(
    logger,
    ct_volumes: torch.Tensor,
    attention_maps: torch.Tensor,
    masks: torch.Tensor,
    stage: str,
    epoch: int,
    max_samples: int = 4
):
    """
    Log localization visualizations to WandB.

    Args:
        logger: WandB logger instance
        ct_volumes: Batch of CT volumes (B, 1, D, H, W)
        attention_maps: Batch of attention maps (B, 1, D', H', W')
        masks: Batch of ground truth masks (B, 1, D, H, W)
        stage: 'train' or 'val'
        epoch: Current epoch number
        max_samples: Maximum number of samples to visualize (default: 4)
    """
    if logger is None:
        return

    # Only visualize samples with annotations
    has_annotation = masks.sum(dim=(2, 3, 4)) > 0  # (B, 1)
    annotated_indices = torch.where(has_annotation.squeeze())[0]

    if len(annotated_indices) == 0:
        return

    # Limit to max_samples
    num_to_log = min(max_samples, len(annotated_indices))
    sample_indices = annotated_indices[:num_to_log]

    visualizations = []
    for idx in sample_indices:
        ct_vol = ct_volumes[idx:idx+1]  # (1, 1, D, H, W)
        attn = attention_maps[idx:idx+1]  # (1, 1, D', H', W')
        mask = masks[idx:idx+1]  # (1, 1, D, H, W)

        vis = create_localization_visualization(
            ct_vol.squeeze(0),  # (1, D, H, W)
            attn.squeeze(0),    # (1, D', H', W')
            mask.squeeze(0),    # (1, D, H, W)
            num_slices=3
        )
        visualizations.append(vis)

    # Log to WandB
    if hasattr(logger, 'experiment'):
        logger.experiment.log({
            f'{stage}_localization_viz': visualizations,
            'epoch': epoch
        })
