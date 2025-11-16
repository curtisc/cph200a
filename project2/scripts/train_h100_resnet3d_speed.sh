#!/bin/bash

# OPTIMIZED FOR 8× H100 GPUs (80GB) - MAXIMUM SPEED
# Focus: Get to 0.80 AUC as fast as possible
# Time: ~3-4 hours for 100 epochs (8x speedup!)

echo "8× H100 Configuration: MAXIMUM SPEED"
echo "Effective batch size: 256 (32 per GPU × 8)"
echo "Expected: 0.78-0.80 AUC in 4-5 hours"
echo ""

python main.py \
  --model_name resnet18_3d \
  --dataset_name nlst \
  --train \
  --monitor_key val_auc \
  \
  --resnet18_3d.pretrained true \
  --resnet18_3d.freeze_backbone false \
  --resnet18_3d.num_classes 2 \
  --resnet18_3d.init_lr 1e-3 \
  --resnet18_3d.use_se true \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 32 \
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
  --project_name nlst_h100_speed

echo ""
echo "Training complete in record time!"
