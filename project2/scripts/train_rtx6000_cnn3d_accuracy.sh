#!/bin/bash

# OPTIMIZED FOR 2× RTX 6000 Pro GPUs (96GB) - MAXIMUM ACCURACY (CNN3D)
# Hardware: 16-core CPU, 128GB RAM, PCIe x8 per GPU (no NVLink)
# Model: CNN3D (simple 3D CNN with deeper architecture)
# Focus: Achieve highest possible AUC (0.78-0.82+)
# Full resolution, all slices, optimal batch size
# Time: ~24-32 hours for 100 epochs
#
# Key optimizations for this hardware:
# - num_workers=10 (limited by 128GB RAM)
# - NCCL flags for PCIe without NVLink
# - Full 256×256×200 resolution

echo "2× RTX 6000 Pro Configuration: CNN3D MAXIMUM ACCURACY"
echo "Full resolution: 256×256×200 (no downsampling!)"
echo "Effective batch size: 4 (2 per GPU × 2)"
echo "Workers: 10 total (RAM-limited)"
echo "Expected: 0.78-0.82 AUC in 24-32 hours"
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
  --cnn3d.num_layers 4 \
  --cnn3d.num_classes 2 \
  --cnn3d.use_bn true \
  --cnn3d.init_lr 5e-4 \
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
  --project_name nlst_rtx6000_cnn3d_accuracy

echo ""
echo "Maximum accuracy CNN3D training complete!"
echo "This config should achieve competitive performance."
