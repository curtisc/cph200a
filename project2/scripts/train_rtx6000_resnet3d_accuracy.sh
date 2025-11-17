#!/bin/bash

# OPTIMIZED FOR 2× RTX 6000 Pro GPUs (96GB) - MAXIMUM ACCURACY (ResNet18_3D)
# Hardware: 16-core CPU, 128GB RAM, PCIe x8 per GPU (no NVLink)
# Model: ResNet18_3D (3D ResNet with SE blocks)
# Focus: Achieve highest possible AUC (0.82-0.85+)
# Full resolution, all slices, optimal batch size
# Time: ~32-40 hours for 100 epochs
#
# Key optimizations for this hardware:
# - num_workers=10 (limited by 128GB RAM)
# - NCCL flags for PCIe without NVLink
# - Batch size optimized for larger model

echo "2× RTX 6000 Pro Configuration: ResNet18_3D MAXIMUM ACCURACY"
echo "Full resolution: 256×256×200 (no downsampling!)"
echo "Effective batch size: 6 (3 per GPU × 2)"
echo "Workers: 10 total (RAM-limited)"
echo "Expected: 0.82-0.85 AUC in 32-40 hours"
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
  --resnet18_3d.init_lr 5e-4 \
  --resnet18_3d.use_se true \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 3 \
  --nlst.num_workers 10 \
  --nlst.img_size "[256, 256]" \
  --nlst.num_images 200 \
  \
  --trainer.max_epochs 100 \
  --trainer.precision bf16-mixed \
  --trainer.gradient_clip_val 1.0 \
  --trainer.devices 2 \
  --trainer.strategy ddp \
  --trainer.log_every_n_steps 5 \
  \
  --project_name nlst_rtx6000_resnet3d_accuracy

echo ""
echo "Maximum accuracy ResNet18_3D training complete!"
echo "This config should achieve state-of-the-art performance."
