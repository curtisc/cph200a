#!/bin/bash

# OPTIMIZED FOR 8× H100 GPUs (80GB) - ULTRA HIGH CAPACITY (CNN3D)
# Model: CNN3D with ultra-deep architecture and high resolution
# Focus: Push the limits - maximum resolution and depth
# Beyond clinical resolution - for research/publications
# Time: ~12-15 hours for 100 epochs

echo "8× H100 Configuration: CNN3D ULTRA HIGH CAPACITY"
echo "Ultra-high resolution: 384×384×224 slices"
echo "Effective batch size: 16 (2 per GPU × 8)"
echo "Architecture: 5 layers with 256-512 channel progression"
echo "Expected: 0.80-0.84+ AUC (potential state-of-the-art for simple CNN)"
echo ""

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
  --nlst.batch_size 2 \
  --nlst.num_workers 8 \
  --nlst.img_size "[384, 384]" \
  --nlst.num_images 224 \
  \
  --trainer.max_epochs 100 \
  --trainer.precision bf16-mixed \
  --trainer.gradient_clip_val 1.0 \
  --trainer.devices 8 \
  --trainer.strategy ddp \
  --trainer.log_every_n_steps 5 \
  \
  --project_name nlst_h100_cnn3d_ultra

echo ""
echo "Ultra high-capacity CNN3D training complete!"
echo "This represents maximum capacity for a simple 3D CNN architecture."
