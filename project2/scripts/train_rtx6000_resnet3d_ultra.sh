#!/bin/bash

# OPTIMIZED FOR 2× RTX 6000 Pro GPUs (96GB) - ULTRA HIGH CAPACITY (ResNet18_3D)
# Hardware: 16-core CPU, 128GB RAM, PCIe x8 per GPU (no NVLink)
# Model: ResNet18_3D (3D ResNet with SE blocks)
# Focus: Push the limits of 96GB VRAM for research purposes
# Ultra-high resolution: 384×384×224 slices
# Time: ~60-80 hours for 100 epochs
#
# Key optimizations for this hardware:
# - num_workers=10 (limited by 128GB RAM)
# - NCCL flags for PCIe without NVLink
# - Larger batch leveraging 96GB VRAM

echo "2× RTX 6000 Pro Configuration: ResNet18_3D ULTRA HIGH CAPACITY"
echo "Ultra resolution: 384×384×224 slices"
echo "Effective batch size: 4 (2 per GPU × 2)"
echo "Workers: 10 total (RAM-limited)"
echo "Expected: 0.83-0.86 AUC in 60-80 hours"
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
  --resnet18_3d.init_lr 3e-4 \
  --resnet18_3d.use_se true \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 2 \
  --nlst.num_workers 10 \
  --nlst.img_size "[384, 384]" \
  --nlst.num_images 224 \
  \
  --trainer.max_epochs 100 \
  --trainer.precision bf16-mixed \
  --trainer.gradient_clip_val 1.0 \
  --trainer.devices 2 \
  --trainer.strategy ddp \
  --trainer.log_every_n_steps 5 \
  \
  --project_name nlst_rtx6000_resnet3d_ultra

echo ""
echo "Ultra high-capacity ResNet18_3D training complete!"
echo "This pushes the limits of 96GB VRAM per GPU."
