#!/bin/bash

# OPTIMIZED FOR 8× H100 GPUs (80GB) - ATTENTION + LOCALIZATION (Video ResNet)
# Model: ResNet18Video3D with attention pooling and localization regularization
# Focus: Achieve highest possible AUC using advanced features
# Time: ~8-10 hours for 100 epochs

echo "8× H100 Configuration: ResNet18Video3D + ATTENTION + LOCALIZATION"
echo "Full resolution: 256×256×200 (no downsampling!)"
echo "Effective batch size: 64 (8 per GPU × 8)"
echo "Features: Spatial attention pooling + Localization regularization"
echo "Expected: 0.82-0.85+ AUC in 8-10 hours"
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
  --resnet18_video3d.init_lr 5e-4 \
  --resnet18_video3d.use_attention_pooling true \
  --resnet18_video3d.use_localization_reg true \
  --resnet18_video3d.localization_reg_weight 0.1 \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 8 \
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
  --project_name nlst_h100_video3d_attention_loc

echo ""
echo "Video ResNet with attention + localization training complete!"
echo "This config should achieve state-of-the-art performance."
