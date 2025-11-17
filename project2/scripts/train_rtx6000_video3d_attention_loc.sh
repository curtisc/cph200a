#!/bin/bash

# OPTIMIZED FOR 2× RTX 6000 Pro GPUs (96GB) - ATTENTION + LOCALIZATION (ResNet18Video3D)
# Hardware: 16-core CPU, 128GB RAM, PCIe x8 per GPU (no NVLink)
# Model: ResNet18Video3D with attention pooling + localization regularization
# Focus: Improve attention map quality using localization masks
# Full resolution: 256×256×200
# Time: ~32-40 hours for 100 epochs
#
# Key optimizations for this hardware:
# - num_workers=10 (limited by 128GB RAM)
# - NCCL flags for PCIe without NVLink
# - Localization regularization weight: 0.1

echo "2× RTX 6000 Pro Configuration: ResNet18Video3D ATTENTION + LOCALIZATION"
echo "Full resolution: 256×256×200"
echo "Features: Attention pooling + localization regularization"
echo "Effective batch size: 4 (2 per GPU × 2)"
echo "Workers: 10 total (RAM-limited)"
echo "Expected: 0.82-0.85 AUC in 32-40 hours"
echo ""

# CRITICAL: Configure NCCL for PCIe without NVLink
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  --resnet18_video3d.localization_reg_weight 0.1 \
  --resnet18_video3d.init_lr 5e-4 \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 2 \
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
  --project_name nlst_rtx6000_video3d_attention_loc

echo ""
echo "ResNet18Video3D with attention + localization training complete!"
echo "Localization regularization improves attention quality."
