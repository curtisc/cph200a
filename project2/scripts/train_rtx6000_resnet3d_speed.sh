#!/bin/bash

# OPTIMIZED FOR 2× RTX 6000 Pro GPUs (96GB) - MAXIMUM SPEED (ResNet18_3D)
# Hardware: 16-core CPU, 128GB RAM, PCIe x8 per GPU (no NVLink)
# Model: ResNet18_3D (3D ResNet with SE blocks)
# Focus: Get to 0.80+ AUC as fast as possible
# Time: ~12-16 hours for 100 epochs
#
# Key optimizations for this hardware:
# - num_workers=10 (limited by 128GB RAM)
# - NCCL flags for PCIe without NVLink
# - Larger batch size leveraging 96GB VRAM

echo "2× RTX 6000 Pro Configuration: ResNet18_3D MAXIMUM SPEED"
echo "Reduced resolution: 128×128×96 slices"
echo "Effective batch size: 24 (12 per GPU × 2)"
echo "Workers: 10 total (RAM-limited)"
echo "Expected: 0.78-0.80 AUC in 12-16 hours"
echo ""

# CRITICAL: Configure NCCL for PCIe without NVLink
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --model_name resnet18_3d \
  --dataset_name nlst \
  --train \
  --monitor_key val_auc \
  \
  --resnet18_3d.pretrained true \
  --resnet18_3d.freeze_backbone false \
  --resnet18_3d.num_classes 2 \
  --resnet18_3d.init_lr 1e-3 \
  --resnet18_3d.use_se true \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 12 \
  --nlst.num_workers 10 \
  --nlst.img_size "[128, 128]" \
  --nlst.num_images 96 \
  \
  --trainer.max_epochs 100 \
  --trainer.precision bf16-mixed \
  --trainer.gradient_clip_val 1.0 \
  --trainer.devices 2 \
  --trainer.strategy ddp \
  --trainer.log_every_n_steps 5 \
  \
  --project_name nlst_rtx6000_resnet3d_speed

echo ""
echo "ResNet18_3D speed training complete!"
