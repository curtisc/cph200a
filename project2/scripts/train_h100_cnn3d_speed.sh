#!/bin/bash

# OPTIMIZED FOR 8× H100 GPUs (80GB) - MAXIMUM SPEED (CNN3D)
# Model: CNN3D (simple 3D CNN)
# Focus: Get to 0.78+ AUC as fast as possible
# Time: ~2-3 hours for 100 epochs

echo "8× H100 Configuration: CNN3D MAXIMUM SPEED"
echo "Reduced resolution: 128×128×96 slices"
echo "Effective batch size: 128 (16 per GPU × 8)"
echo "Expected: 0.75-0.78 AUC in 2-3 hours"
echo ""

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
  --nlst.batch_size 16 \
  --nlst.num_workers 8 \
  --nlst.img_size "[128, 128]" \
  --nlst.num_images 96 \
  \
  --trainer.max_epochs 100 \
  --trainer.precision bf16-mixed \
  --trainer.gradient_clip_val 1.0 \
  --trainer.devices 8 \
  --trainer.strategy ddp \
  --trainer.log_every_n_steps 5 \
  \
  --project_name nlst_h100_cnn3d_speed

echo ""
echo "CNN3D speed training complete!"
