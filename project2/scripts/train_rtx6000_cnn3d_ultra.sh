#!/bin/bash

# OPTIMIZED FOR 2× RTX 6000 Pro GPUs (96GB) - ULTRA HIGH CAPACITY (CNN3D)
# Hardware: 16-core CPU, 128GB RAM, PCIe x8 per GPU (no NVLink)
# Model: CNN3D (deeper architecture, maximum resolution)
# Focus: Push the limits of 96GB VRAM for research purposes
# Ultra-high resolution: 384×384×224 slices
# Time: ~48-60 hours for 100 epochs
#
# Key optimizations for this hardware:
# - num_workers=10 (limited by 128GB RAM)
# - NCCL flags for PCIe without NVLink
# - batch_size=1 per GPU (memory limit)

echo "2× RTX 6000 Pro Configuration: CNN3D ULTRA HIGH CAPACITY"
echo "Ultra resolution: 384×384×224 slices"
echo "Effective batch size: 2 (1 per GPU × 2)"
echo "Workers: 10 total (RAM-limited)"
echo "Expected: 0.80-0.84 AUC in 48-60 hours"
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
  --cnn3d.hidden_dim 256 \
  --cnn3d.num_layers 5 \
  --cnn3d.num_classes 2 \
  --cnn3d.use_bn true \
  --cnn3d.init_lr 3e-4 \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 1 \
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
  --project_name nlst_rtx6000_cnn3d_ultra

echo ""
echo "Ultra high-capacity CNN3D training complete!"
echo "This pushes the limits of 96GB VRAM per GPU."
