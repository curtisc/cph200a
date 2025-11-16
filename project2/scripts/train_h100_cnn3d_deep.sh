#!/bin/bash

# OPTIMIZED FOR 8× H100 GPUs (80GB) - DEEP ARCHITECTURE (CNN3D)
# Model: CNN3D with deeper architecture (5 layers)
# Focus: Maximize capacity with deeper network
# Time: ~6-8 hours for 100 epochs

echo "8× H100 Configuration: CNN3D DEEP ARCHITECTURE"
echo "Full resolution: 256×256×200"
echo "Effective batch size: 32 (4 per GPU × 8)"
echo "Architecture: 5 convolutional layers with increased capacity"
echo "Expected: 0.79-0.83 AUC in 6-8 hours"
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
  --project_name nlst_h100_cnn3d_deep

echo ""
echo "Deep CNN3D training complete!"
echo "This deeper architecture should provide improved feature learning."
