#!/bin/bash

# FIXED V2 FOR 2× RTX 6000 Pro GPUs - ATTENTION + LOCALIZATION
#
# FIXES APPLIED:
# 1. Increased batch size from 2 to 4 (minimum safe for BatchNorm3d)
# 2. Reduced image resolution to 128x128x100 to fit in memory
# 3. Fixed KL divergence numerical stability (log(0) = -inf bug)
# 4. Added gradient clipping safety
# 5. Lower learning rate for stability
# 6. Removed old checkpoints to force fresh start
#
# This version includes the CRITICAL KL divergence fix that prevents log(0) = -inf

echo "2× RTX 6000 Pro Configuration: ResNet18Video3D ATTENTION + LOCALIZATION (FIXED V2)"
echo "Resolution: 128×128×100 (downsampled for stability)"
echo "Features: Attention pooling + localization regularization"
echo "Effective batch size: 32 (4 per GPU × 2 GPUs × 4 accumulation)"
echo "Workers: 8"
echo "Expected: 0.80-0.84 AUC in 20-25 hours"
echo ""
echo "IMPORTANT: KL divergence numerical stability bug has been FIXED!"
echo ""

# Clean up old checkpoints to ensure fresh start
echo "Removing old checkpoints from failed runs..."
rm -rf scripts/nlst_rtx6000_video3d_attention_loc_fixed
echo "Starting fresh training run..."
echo ""

# CRITICAL: Configure NCCL for PCIe without NVLink
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable anomaly detection to catch NaN early
export PYTORCH_ENABLE_ANOMALY_DETECTION=1

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
  --resnet18_video3d.localization_reg_weight 0.05 \
  --resnet18_video3d.init_lr 2e-4 \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 4 \
  --nlst.num_workers 8 \
  --nlst.img_size "[128, 128]" \
  --nlst.num_images 100 \
  \
  --trainer.max_epochs 100 \
  --trainer.precision bf16-mixed \
  --trainer.gradient_clip_val 0.5 \
  --trainer.accumulate_grad_batches 4 \
  --trainer.devices 2 \
  --trainer.strategy ddp \
  --trainer.log_every_n_steps 5 \
  \
  --project_name nlst_rtx6000_video3d_attention_loc_fixed_v2

echo ""
echo "Training complete!"
echo ""
echo "What to expect:"
echo "  - Epoch 1: train_auc should rise from 0.5 to 0.55-0.60 (NO NaN!)"
echo "  - Epoch 5: train_auc should be 0.65-0.75"
echo "  - Epoch 20+: val_auc should be 0.80-0.84"
echo ""
echo "If train_auc is still 0.5 after epoch 3, check for NaN in logs!"
