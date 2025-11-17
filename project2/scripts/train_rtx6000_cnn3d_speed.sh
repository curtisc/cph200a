#!/bin/bash

# OPTIMIZED FOR 2× RTX 6000 Pro GPUs (96GB) - MAXIMUM SPEED (CNN3D)
# Hardware: 16-core CPU, 128GB RAM, PCIe x8 per GPU (no NVLink)
# Model: CNN3D (simple 3D CNN)
# Focus: Get to 0.78+ AUC as fast as possible
# Time: ~8-12 hours for 100 epochs
#
# Key optimizations for this hardware:
# - num_workers=10 (limited by 128GB RAM, not per-GPU)
# - NCCL flags for PCIe without NVLink
# - Batch size optimized for 96GB VRAM

echo "2× RTX 6000 Pro Configuration: CNN3D MAXIMUM SPEED"
echo "Reduced resolution: 128×128×96 slices"
echo "Effective batch size: 16 (8 per GPU × 2)"
echo "Workers: 10 total (RAM-limited)"
echo "Expected: 0.75-0.78 AUC in 8-12 hours"
echo ""

# CRITICAL: Configure NCCL for PCIe without NVLink
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --model_name cnn3d \
  --dataset_name nlst \
  --train \
  --monitor_key val_auc \
  \
  --cnn3d.input_channels 1 \
  --cnn3d.hidden_dim 128 \
  --cnn3d.num_layers 3 \
  --cnn3d.num_classes 2 \
  --cnn3d.use_bn true \
  --cnn3d.init_lr 1e-3 \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 8 \
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
  --project_name nlst_rtx6000_cnn3d_speed

echo ""
echo "CNN3D speed training complete!"
