#!/bin/bash

# Quick test with 10% of data to verify improvements
# Should show ~0.70+ AUC if fixes are working

echo "Running quick test with 10% of data to verify optimizations..."
echo "Expected AUC: 0.68-0.72 (up from 0.649)"
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
  --nlst.data_percent 10 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 20 \
  --nlst.num_workers 4 \
  --nlst.img_size "[128, 128]" \
  --nlst.num_images 96 \
  \
  --trainer.max_epochs 30 \
  --trainer.precision bf16-mixed \
  --trainer.gradient_clip_val 1.0 \
  --trainer.devices 2 \
  --trainer.strategy ddp \
  \
  --project_name nlst_quick_test

echo ""
echo "Quick test complete!"
echo "If val_auc > 0.70, the optimizations are working correctly."
echo "Run train_optimal.sh with 100% data to reach 0.80+ AUC"
