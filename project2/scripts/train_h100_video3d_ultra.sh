#!/bin/bash

# OPTIMIZED FOR 8× H100 GPUs (80GB) - ULTRA HIGH CAPACITY (Video ResNet)
# Model: ResNet18Video3D with attention and localization
# Focus: Push the limits - maximum resolution and depth
# Beyond clinical resolution - for research/publications
# Time: ~15-20 hours for 100 epochs

echo "8× H100 Configuration: ResNet18Video3D ULTRA HIGH CAPACITY"
echo "Ultra-high resolution: 384×384×224 slices"
echo "Effective batch size: 24 (3 per GPU × 8)"
echo "Features: Attention pooling + Localization regularization"
echo "Expected: 0.83-0.86+ AUC (potential state-of-the-art)"
echo "WARNING: This is pushing H100 80GB limits for research purposes"
echo ""

python main.py \
  --model_name resnet18_video3d \
  --dataset_name nlst \
  --train \
  --monitor_key val_auc \
  \
  --resnet18_video3d.pretrained true \
  --resnet18_video3d.freeze_backbone false \
  --resnet18_video3d.num_classes 2 \
  --resnet18_video3d.init_lr 3e-4 \
  --resnet18_video3d.use_attention_pooling true \
  --resnet18_video3d.use_localization_reg true \
  --resnet18_video3d.localization_reg_weight 0.1 \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 3 \
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
  --project_name nlst_h100_video3d_ultra

echo ""
echo "Ultra high-capacity Video ResNet training complete!"
echo "This represents the upper limit of what's computationally feasible on H100 80GB."
