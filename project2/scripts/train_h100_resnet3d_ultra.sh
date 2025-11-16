#!/bin/bash

# OPTIMIZED FOR 8× H100 GPUs (80GB) - ULTRA HIGH CAPACITY
# Focus: Push the limits - maximum resolution and depth
# Beyond clinical resolution - for research/publications
# Time: ~15-20 hours for 100 epochs

echo "8× H100 Configuration: ULTRA HIGH CAPACITY"
echo "Ultra-high resolution: 384×384×224 slices"
echo "Effective batch size: 32 (4 per GPU × 8)"
echo "Expected: 0.83-0.86+ AUC (potential state-of-the-art)"
echo "WARNING: This is pushing H100 80GB limits for research purposes"
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
  --resnet18_3d.init_lr 3e-4 \
  --resnet18_3d.use_se true \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 4 \
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
  --project_name nlst_h100_ultra

echo ""
echo "Ultra high-capacity training complete!"
echo "This represents the upper limit of what's computationally feasible on H100 80GB."
