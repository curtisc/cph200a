#!/bin/bash

# OPTIMIZED FOR 8× H100 GPUs (80GB) - ATTENTION + LOCALIZATION (CNN3D)
# Model: CNN3D with attention pooling and localization regularization
# Focus: Achieve highest possible AUC using advanced features
# Time: ~6-8 hours for 100 epochs

echo "8× H100 Configuration: CNN3D + ATTENTION + LOCALIZATION"
echo "Full resolution: 256×256×200 (no downsampling!)"
echo "Effective batch size: 32 (4 per GPU × 8)"
echo "Features: Spatial attention pooling + Localization regularization"
echo "Expected: 0.80-0.83+ AUC in 6-8 hours"
echo ""

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
  --cnn3d.use_attention_pooling true \
  --cnn3d.use_localization_reg true \
  --cnn3d.localization_reg_weight 0.1 \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 4 \
  --nlst.num_workers 8 \
  --nlst.img_size "[256, 256]" \
  --nlst.num_images 200 \
  \
  --trainer.max_epochs 100 \
  --trainer.precision bf16-mixed \
  --trainer.gradient_clip_val 1.0 \
  --trainer.devices 8 \
  --trainer.strategy ddp \
  --trainer.log_every_n_steps 5 \
  \
  --project_name nlst_h100_cnn3d_attention_loc

echo ""
echo "CNN3D with attention + localization training complete!"
echo "This config should achieve improved performance over baseline."
