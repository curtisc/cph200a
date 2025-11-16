#!/bin/bash

# OPTIMIZED FOR 8× H100 GPUs (80GB) - LOCALIZATION REGULARIZATION ONLY
# Focus: Test localization regularization with standard average pooling
# Uses localization masks to guide feature learning without attention
# Time: ~8-10 hours for 100 epochs

echo "8× H100 Configuration: LOCALIZATION REGULARIZATION (NO ATTENTION)"
echo "Full resolution: 256×256×200 (no downsampling!)"
echo "Effective batch size: 96 (12 per GPU × 8)"
echo "Features: Localization regularization with avg pooling"
echo "Expected: 0.80-0.83 AUC in 8-10 hours"
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
  --resnet18_3d.use_attention_pooling false \
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
  --project_name nlst_h100_localization_only

echo ""
echo "Localization regularization training complete!"
