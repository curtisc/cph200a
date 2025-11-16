#!/bin/bash

# HIGH-RESOLUTION training for maximum accuracy
# Uses your hardware to its fullest: 256x256 resolution, 128 slices
# Expected: 0.80-0.83 AUC (higher than standard config)
# Time: ~40-50 hours with 100% data

echo "Starting HIGH-RESOLUTION training..."
echo "Expected: 0.80-0.83 AUC"
echo "Estimated time: 40-50 hours for 100 epochs"
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
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 10 \
  --nlst.num_workers 4 \
  --nlst.img_size "[256, 256]" \
  \
  --trainer.max_epochs 100 \
  --trainer.precision bf16-mixed \
  --trainer.gradient_clip_val 1.0 \
  --trainer.devices 2 \
  --trainer.strategy ddp \
  --trainer.log_every_n_steps 10 

echo ""
echo "High-resolution training complete!"
echo "This configuration should achieve the highest AUC."
