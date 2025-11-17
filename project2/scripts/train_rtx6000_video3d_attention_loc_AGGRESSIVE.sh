#!/bin/bash

# AGGRESSIVE FIX FOR 2× RTX 6000 Pro - Match H100 performance at full resolution
#
# This version attempts to keep full 256×256×200 resolution by:
# 1. Using MUCH more aggressive gradient accumulation (32 steps)
# 2. Minimum safe batch size (4) to prevent BatchNorm collapse
# 3. Lower learning rate to compensate for effective batch=256
# 4. GroupNorm instead of SyncBatchNorm for better small-batch stability
#
# WARNING: This will be VERY SLOW but should match H100 quality
# Time estimate: 80-100 hours for 100 epochs

echo "2× RTX 6000 Pro AGGRESSIVE Configuration: Full Resolution Match"
echo "Resolution: 256×256×200 (FULL RESOLUTION!)"
echo "Effective batch size: 256 (4 per GPU × 2 GPUs × 32 accumulation)"
echo "Features: Attention pooling + localization regularization"
echo "Expected: 0.88-0.90 AUC in 80-100 hours"
echo ""
echo "⚠️  WARNING: This will be very slow but should match H100 performance"
echo ""

# CRITICAL: Configure NCCL for PCIe
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --model_name resnet18_video3d \
  --dataset_name nlst \
  --train \
  --monitor_key val_auc \
  \
  --resnet18_video3d.pretrained true \
  --resnet18_video3d.freeze_backbone false \
  --resnet18_video3d.num_classes 2 \
  --resnet18_video3d.use_attention_pooling true \
  --resnet18_video3d.use_localization_reg true \
  --resnet18_video3d.localization_reg_weight 0.1 \
  --resnet18_video3d.init_lr 2e-4 \
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
  --trainer.gradient_clip_val 0.5 \
  --trainer.accumulate_grad_batches 32 \
  --trainer.devices 2 \
  --trainer.strategy ddp \
  --trainer.log_every_n_steps 5 \
  \
  --project_name nlst_rtx6000_video3d_attention_loc_fullres

echo ""
echo "Full resolution training complete!"
