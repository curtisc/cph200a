#!/bin/bash

# OPTIMIZED FOR 8× H100 GPUs (80GB) - MAXIMUM ACCURACY WITH ATTENTION & LOCALIZATION
# Focus: Achieve highest possible AUC (0.82-0.85+) using attention pooling and localization regularization
# Full resolution, all slices, optimal batch size
# Time: ~8-10 hours for 100 epochs

echo "8× H100 Configuration: MAXIMUM ACCURACY + ATTENTION + LOCALIZATION"
echo "Full resolution: 256×256×200 (no downsampling!)"
echo "Effective batch size: 96 (12 per GPU × 8)"
echo "Features: Spatial attention pooling + Localization regularization"
echo "Expected: 0.82-0.85+ AUC in 8-10 hours"
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
  --resnet18_3d.init_lr 5e-4 \
  --resnet18_3d.use_se true \
  --resnet18_3d.use_attention_pooling true \
  --resnet18_3d.use_localization_reg true \
  --resnet18_3d.localization_reg_weight 0.1 \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 12 \
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
  --project_name nlst_h100_accuracy_v2

echo ""
echo "Maximum accuracy training with attention + localization complete!"
echo "This config should achieve state-of-the-art performance."
