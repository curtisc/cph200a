#!/bin/bash

# =============================================================================
# FAST MINIMAL TRAINING SCRIPT FOR 8× B200 GPUs
# =============================================================================
# Purpose: Train CNN3D, ResNet18, and Video3D as fast as possible
# - No SE blocks
# - No data augmentation
# - No pretraining
# - No localization regularization
# =============================================================================

set -e  # Exit on error

# NCCL Configuration for B200 with NVLink
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
DEVICES=8
PRECISION="bf16-mixed"
NUM_WORKERS=24
MAX_EPOCHS=10
GRADIENT_CLIP=1.0

# Resolution and slices
IMG_SIZE="[256, 256]"
NUM_SLICES=200

# Higher batch sizes since we removed augmentation overhead
BATCH_CNN3D=6      # 6 per GPU = 48 total
BATCH_RESNET3D=4   # 4 per GPU = 32 total
BATCH_VIDEO3D=3    # 3 per GPU = 24 total

# Data paths
DATA_ROOT=~/cphdata
NLST_DIR="${DATA_ROOT}/compressed"
NLST_METADATA="${DATA_ROOT}/nlst-metadata/full_nlst_google.json"
VALID_EXAM_PATH="${DATA_ROOT}/nlst-metadata/valid_exams.p"
LUNGRADS_PATH="${DATA_ROOT}/nlst-metadata/nlst_acc2lungrads.p"

echo "============================================================================="
echo "B200 FAST MINIMAL TRAINING"
echo "============================================================================="
echo "Models: CNN3D, ResNet18, Video3D"
echo "Disabled: SE blocks, augmentation, pretraining, localization"
echo "Batch sizes: CNN3D=${BATCH_CNN3D}, ResNet3D=${BATCH_RESNET3D}, Video3D=${BATCH_VIDEO3D}"
echo "============================================================================="
echo ""

# =============================================================================
# EXPERIMENT 1: Simple 3D CNN
# =============================================================================
train_cnn3d() {
    echo ""
    echo "============================================================================="
    echo "Training: Simple 3D CNN"
    echo "============================================================================="

    python main.py \
      --model_name cnn3d \
      --dataset_name nlst \
      --train \
      --monitor_key val_auc \
      \
      --cnn3d.input_channels 1 \
      --cnn3d.hidden_dim 256 \
      --cnn3d.num_layers 5 \
      --cnn3d.num_classes 2 \
      --cnn3d.use_bn true \
      --cnn3d.init_lr 1e-3 \
      --cnn3d.use_attention_pooling false \
      \
      --nlst.nlst_dir "${NLST_DIR}" \
      --nlst.nlst_metadata_path "${NLST_METADATA}" \
      --nlst.valid_exam_path "${VALID_EXAM_PATH}" \
      --nlst.lungrads_path "${LUNGRADS_PATH}" \
      --nlst.data_percent 100 \
      --nlst.use_data_augmentation false \
      --nlst.class_balance true \
      --nlst.batch_size ${BATCH_CNN3D} \
      --nlst.num_workers ${NUM_WORKERS} \
      --nlst.img_size "${IMG_SIZE}" \
      --nlst.num_images ${NUM_SLICES} \
      \
      --trainer.max_epochs ${MAX_EPOCHS} \
      --trainer.precision ${PRECISION} \
      --trainer.gradient_clip_val ${GRADIENT_CLIP} \
      --trainer.devices ${DEVICES} \
      --trainer.strategy ddp \
      --trainer.log_every_n_steps 5 \
      \
      --project_name nlst_b200_cnn3d_fast

    echo "CNN3D training complete!"
}

# =============================================================================
# EXPERIMENT 2: ResNet-18 3D (From Scratch, No SE)
# =============================================================================
train_resnet3d() {
    echo ""
    echo "============================================================================="
    echo "Training: ResNet-18 3D (no pretrain, no SE)"
    echo "============================================================================="

    python main.py \
      --model_name resnet18_3d \
      --dataset_name nlst \
      --train \
      --monitor_key val_auc \
      \
      --resnet18_3d.pretrained false \
      --resnet18_3d.freeze_backbone false \
      --resnet18_3d.num_classes 2 \
      --resnet18_3d.init_lr 1e-3 \
      --resnet18_3d.use_se false \
      --resnet18_3d.use_attention_pooling false \
      --resnet18_3d.use_localization_reg false \
      \
      --nlst.nlst_dir "${NLST_DIR}" \
      --nlst.nlst_metadata_path "${NLST_METADATA}" \
      --nlst.valid_exam_path "${VALID_EXAM_PATH}" \
      --nlst.lungrads_path "${LUNGRADS_PATH}" \
      --nlst.data_percent 100 \
      --nlst.use_data_augmentation false \
      --nlst.class_balance true \
      --nlst.batch_size ${BATCH_RESNET3D} \
      --nlst.num_workers ${NUM_WORKERS} \
      --nlst.img_size "${IMG_SIZE}" \
      --nlst.num_images ${NUM_SLICES} \
      \
      --trainer.max_epochs ${MAX_EPOCHS} \
      --trainer.precision ${PRECISION} \
      --trainer.gradient_clip_val ${GRADIENT_CLIP} \
      --trainer.devices ${DEVICES} \
      --trainer.strategy ddp \
      --trainer.log_every_n_steps 5 \
      \
      --project_name nlst_b200_resnet3d_fast

    echo "ResNet-18 3D training complete!"
}

# =============================================================================
# EXPERIMENT 3: Video ResNet-3D (From Scratch)
# =============================================================================
train_video3d() {
    echo ""
    echo "============================================================================="
    echo "Training: Video ResNet-3D (no pretrain)"
    echo "============================================================================="

    python main.py \
      --model_name resnet18_video3d \
      --dataset_name nlst \
      --train \
      --monitor_key val_auc \
      \
      --resnet18_video3d.pretrained false \
      --resnet18_video3d.freeze_backbone false \
      --resnet18_video3d.num_classes 2 \
      --resnet18_video3d.init_lr 1e-3 \
      --resnet18_video3d.use_attention_pooling false \
      --resnet18_video3d.use_localization_reg false \
      \
      --nlst.nlst_dir "${NLST_DIR}" \
      --nlst.nlst_metadata_path "${NLST_METADATA}" \
      --nlst.valid_exam_path "${VALID_EXAM_PATH}" \
      --nlst.lungrads_path "${LUNGRADS_PATH}" \
      --nlst.data_percent 100 \
      --nlst.use_data_augmentation false \
      --nlst.class_balance true \
      --nlst.batch_size ${BATCH_VIDEO3D} \
      --nlst.num_workers ${NUM_WORKERS} \
      --nlst.img_size "${IMG_SIZE}" \
      --nlst.num_images ${NUM_SLICES} \
      \
      --trainer.max_epochs ${MAX_EPOCHS} \
      --trainer.precision ${PRECISION} \
      --trainer.gradient_clip_val ${GRADIENT_CLIP} \
      --trainer.devices ${DEVICES} \
      --trainer.strategy ddp \
      --trainer.log_every_n_steps 5 \
      \
      --project_name nlst_b200_video3d_fast

    echo "Video ResNet-3D training complete!"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "Starting training sequence..."
echo ""

train_cnn3d
train_resnet3d
train_video3d

echo ""
echo "============================================================================="
echo "ALL TRAINING COMPLETE!"
echo "============================================================================="
echo "Results saved in models/ directory:"
echo "  - nlst_b200_cnn3d_fast"
echo "  - nlst_b200_resnet3d_fast"
echo "  - nlst_b200_video3d_fast"
echo ""
