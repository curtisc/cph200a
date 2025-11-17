#!/bin/bash

# OPTIMIZED FOR 2× RTX 6000 Pro GPUs (96GB) - ULTRA HIGH CAPACITY (ResNet18Video3D)
# Hardware: 16-core CPU, 128GB RAM, PCIe x8 per GPU (no NVLink)
# Model: ResNet18Video3D (torchvision r3d_18, Kinetics-400 pretrained)
# Focus: Push the limits of 96GB VRAM for research purposes
# Ultra-high resolution: 384×384×224 slices
# Time: ~60-80 hours for 100 epochs
#
# Key optimizations for this hardware:
# - num_workers=10 (limited by 128GB RAM)
# - NCCL flags for PCIe without NVLink
# - Conservative batch for very large model

echo "2× RTX 6000 Pro Configuration: ResNet18Video3D ULTRA HIGH CAPACITY"
echo "Ultra resolution: 384×384×224 slices"
echo "Effective batch size: 2 (1 per GPU × 2)"
echo "Workers: 10 total (RAM-limited)"
echo "Expected: 0.83-0.86 AUC in 60-80 hours"
echo ""

# CRITICAL: Configure NCCL for PCIe without NVLink
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
  --resnet18_video3d.init_lr 3e-4 \
  \
  --nlst.data_percent 100 \
  --nlst.use_data_augmentation true \
  --nlst.class_balance true \
  --nlst.batch_size 1 \
  --nlst.num_workers 10 \
  --nlst.img_size "[384, 384]" \
  --nlst.num_images 224 \
  \
  --trainer.max_epochs 100 \
  --trainer.precision bf16-mixed \
  --trainer.gradient_clip_val 1.0 \
  --trainer.devices 2 \
  --trainer.strategy ddp \
  --trainer.log_every_n_steps 5 \
  \
  --project_name nlst_rtx6000_video3d_ultra

echo ""
echo "Ultra high-capacity ResNet18Video3D training complete!"
echo "This pushes the limits of 96GB VRAM per GPU."
