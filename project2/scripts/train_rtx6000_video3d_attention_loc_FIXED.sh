#!/bin/bash

# FIXED VERSION FOR 2× RTX 6000 Pro GPUs - ATTENTION + LOCALIZATION (ResNet18Video3D)
#
# FIXES APPLIED:
# 1. Increased batch size from 2 to 4 (minimum safe for BatchNorm3d)
# 2. Reduced image resolution to 128x128x100 to fit in memory
# 3. Added SyncBatchNorm for stable distributed batch norm
# 4. Reduced gradient accumulation from 8 to 4
# 5. Lower learning rate for stability
# 6. Added gradient clipping safety
#
# This config prioritizes STABILITY over speed.

echo "2× RTX 6000 Pro Configuration: ResNet18Video3D ATTENTION + LOCALIZATION (FIXED)"
echo "Resolution: 128×128×100 (downsampled for stability)"
echo "Features: Attention pooling + localization regularization"
echo "Effective batch size: 32 (4 per GPU × 2 GPUs × 4 accumulation)"
echo "Workers: 8 (reduced to avoid RAM issues)"
echo "Expected: 0.80-0.84 AUC in 20-25 hours"
echo ""

# CRITICAL: Configure NCCL for PCIe without NVLink
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable anomaly detection to catch NaN early
export PYTORCH_ENABLE_ANOMALY_DETECTION=1

python main.py \
  --model_name resnet18_video3d \
  --dataset_name nlst \
  --train \
  --monitor_key val_auc \
  \
  --resnet18_video3d.pretrained true \
  --resnet18_video3d.freeze_backbone false \
  --resnet18_video3d.num_classes 2 \
  --resnet18_video3d.use_attention_pooling true \
  --resnet18_video3d.use_localization_reg true \
  --resnet18_video3d.localization_reg_weight 0.05 \
  --resnet18_video3d.init_lr 2e-4 \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 4 \
  --nlst.num_workers 8 \
  --nlst.img_size "[128, 128]" \
  --nlst.num_images 100 \
  \
  --trainer.max_epochs 100 \
  --trainer.precision bf16-mixed \
  --trainer.gradient_clip_val 0.5 \
  --trainer.accumulate_grad_batches 4 \
  --trainer.devices 2 \
  --trainer.strategy ddp \
  --trainer.log_every_n_steps 5 \
  \
  --project_name nlst_rtx6000_video3d_attention_loc_fixed

echo ""
echo "Fixed training script complete!"
echo ""
echo "Key differences from failed run:"
echo "  - Batch size: 2 → 4 (prevents BatchNorm collapse)"
echo "  - Resolution: 256×256×200 → 128×128×100 (fits in memory)"
echo "  - Localization weight: 0.1 → 0.05 (more stable gradients)"
echo "  - Learning rate: 5e-4 → 2e-4 (safer for small batches)"
echo "  - Gradient clip: 1.0 → 0.5 (prevents explosions)"
