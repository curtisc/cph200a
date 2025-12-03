#!/bin/bash

# =============================================================================
# COMPREHENSIVE TRAINING SCRIPT FOR 8× B200 GPUs (180GB VRAM each)
# =============================================================================
# Hardware: 8× B200 GPUs, 208 cores, 2900 GB RAM
# Purpose: Train all models for project report with pretraining analysis
#
# Experiments covered:
# 1. Simple 3D CNN (no pretraining) - baseline
# 2. ResNet-18 3D with ImageNet pretraining
# 3. ResNet-18 3D without ImageNet pretraining (from scratch)
# 4. ResNet-3D Video with Kinetics pretraining
# 5. ResNet-3D Video without pretraining (from scratch)
# 6. Pretraining ablations (frozen backbone, shuffled weights, etc.)
#
# With B200's 180GB VRAM (2.25× H100), we can push resolution significantly higher
# =============================================================================

set -e  # Exit on error

# Configuration
DEVICES=8
PRECISION="bf16-mixed"
NUM_WORKERS=24  # 208 cores / 8 GPUs ≈ 26, use 24 for safety
MAX_EPOCHS=100
GRADIENT_CLIP=1.0

# B200-optimized settings (180GB vs H100's 80GB = 2.25× more VRAM)
# Ultra-high resolution configuration
IMG_SIZE="[512, 512]"
NUM_SLICES=256

# Batch sizes per GPU (can be much larger on B200)
BATCH_CNN3D=6      # CNN3D is memory efficient
BATCH_RESNET3D=8   # ResNet-18 3D
BATCH_VIDEO3D=6    # Video3D slightly larger

echo "============================================================================="
echo "B200 COMPREHENSIVE TRAINING SUITE"
echo "============================================================================="
echo "Hardware: 8× B200 GPUs (180GB each), 208 cores, 2900 GB RAM"
echo "Resolution: 512×512×256 slices (ultra-high)"
echo "Precision: ${PRECISION}"
echo ""
echo "This script will train all models needed for the project report:"
echo "  - Simple 3D CNN"
echo "  - ResNet-18 3D (with/without ImageNet pretraining)"
echo "  - Video ResNet-3D (with/without Kinetics pretraining)"
echo "  - Pretraining ablation experiments"
echo "============================================================================="
echo ""

# =============================================================================
# EXPERIMENT 1: Simple 3D CNN (No Pretraining) - Baseline
# =============================================================================
train_cnn3d() {
    echo ""
    echo "============================================================================="
    echo "EXPERIMENT 1: Simple 3D CNN (No Pretraining)"
    echo "============================================================================="
    echo "Architecture: 5-layer 3D CNN with batch normalization"
    echo "This serves as the baseline without any pretraining"
    echo ""

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
      --cnn3d.init_lr 3e-4 \
      --cnn3d.use_attention_pooling true \
      \
      --nlst.data_percent 100 \
      --nlst.use_data_augmentation true \
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
      --project_name nlst_b200_cnn3d_baseline

    echo "CNN3D baseline training complete!"
}

# =============================================================================
# EXPERIMENT 2a: ResNet-18 3D WITH ImageNet Pretraining
# =============================================================================
train_resnet3d_pretrained() {
    echo ""
    echo "============================================================================="
    echo "EXPERIMENT 2a: ResNet-18 3D WITH ImageNet Pretraining"
    echo "============================================================================="
    echo "Uses ImageNet weights inflated to 3D"
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
      --resnet18_3d.init_lr 3e-4 \
      --resnet18_3d.use_attention_pooling true \
      --resnet18_3d.use_localization_reg true \
      --resnet18_3d.localization_reg_weight 0.1 \
      \
      --nlst.data_percent 100 \
      --nlst.use_data_augmentation true \
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
      --project_name nlst_b200_resnet3d_imagenet_pretrained

    echo "ResNet-18 3D with ImageNet pretraining complete!"
}

# =============================================================================
# EXPERIMENT 2b: ResNet-18 3D WITHOUT Pretraining (From Scratch)
# =============================================================================
train_resnet3d_scratch() {
    echo ""
    echo "============================================================================="
    echo "EXPERIMENT 2b: ResNet-18 3D WITHOUT Pretraining (From Scratch)"
    echo "============================================================================="
    echo "Random initialization - no transfer learning"
    echo ""

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
      --resnet18_3d.use_attention_pooling true \
      --resnet18_3d.use_localization_reg true \
      --resnet18_3d.localization_reg_weight 0.1 \
      \
      --nlst.data_percent 100 \
      --nlst.use_data_augmentation true \
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
      --project_name nlst_b200_resnet3d_from_scratch

    echo "ResNet-18 3D from scratch complete!"
}

# =============================================================================
# EXPERIMENT 3a: Video ResNet-3D WITH Kinetics Pretraining
# =============================================================================
train_video3d_pretrained() {
    echo ""
    echo "============================================================================="
    echo "EXPERIMENT 3a: Video ResNet-3D WITH Kinetics Pretraining"
    echo "============================================================================="
    echo "Uses Kinetics-400 video pretraining weights"
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
      --project_name nlst_b200_video3d_kinetics_pretrained

    echo "Video ResNet-3D with Kinetics pretraining complete!"
}

# =============================================================================
# EXPERIMENT 3b: Video ResNet-3D WITHOUT Pretraining (From Scratch)
# =============================================================================
train_video3d_scratch() {
    echo ""
    echo "============================================================================="
    echo "EXPERIMENT 3b: Video ResNet-3D WITHOUT Pretraining (From Scratch)"
    echo "============================================================================="
    echo "Random initialization - no video pretraining"
    echo ""

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
      --resnet18_video3d.use_attention_pooling true \
      --resnet18_video3d.use_localization_reg true \
      --resnet18_video3d.localization_reg_weight 0.1 \
      \
      --nlst.data_percent 100 \
      --nlst.use_data_augmentation true \
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
      --project_name nlst_b200_video3d_from_scratch

    echo "Video ResNet-3D from scratch complete!"
}

# =============================================================================
# EXPERIMENT 4a: Frozen Backbone (Pure Feature Transfer)
# =============================================================================
# This tests if pretraining helps purely through feature transfer
# If frozen backbone works well, features are directly transferable
train_video3d_frozen() {
    echo ""
    echo "============================================================================="
    echo "EXPERIMENT 4a: Video ResNet-3D with FROZEN Backbone"
    echo "============================================================================="
    echo "Tests pure feature transfer - only classifier is trained"
    echo "If this works well, pretrained features are directly useful"
    echo ""

    python main.py \
      --model_name resnet18_video3d \
      --dataset_name nlst \
      --train \
      --monitor_key val_auc \
      \
      --resnet18_video3d.pretrained true \
      --resnet18_video3d.freeze_backbone true \
      --resnet18_video3d.num_classes 2 \
      --resnet18_video3d.init_lr 1e-3 \
      --resnet18_video3d.use_attention_pooling true \
      --resnet18_video3d.use_localization_reg false \
      \
      --nlst.data_percent 100 \
      --nlst.use_data_augmentation true \
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
      --project_name nlst_b200_video3d_frozen_backbone

    echo "Frozen backbone experiment complete!"
}

# =============================================================================
# EXPERIMENT 4b: ResNet-18 3D Frozen Backbone
# =============================================================================
train_resnet3d_frozen() {
    echo ""
    echo "============================================================================="
    echo "EXPERIMENT 4b: ResNet-18 3D with FROZEN Backbone"
    echo "============================================================================="
    echo "Tests pure feature transfer from ImageNet"
    echo ""

    python main.py \
      --model_name resnet18_3d \
      --dataset_name nlst \
      --train \
      --monitor_key val_auc \
      \
      --resnet18_3d.pretrained true \
      --resnet18_3d.freeze_backbone true \
      --resnet18_3d.num_classes 2 \
      --resnet18_3d.init_lr 1e-3 \
      --resnet18_3d.use_attention_pooling true \
      --resnet18_3d.use_localization_reg false \
      \
      --nlst.data_percent 100 \
      --nlst.use_data_augmentation true \
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
      --project_name nlst_b200_resnet3d_frozen_backbone

    echo "ResNet-18 3D frozen backbone experiment complete!"
}

# =============================================================================
# EXPERIMENT 5: Gradual Unfreezing (Progressive Fine-tuning)
# =============================================================================
# Start with frozen backbone, then unfreeze layers progressively
# This tests the optimal balance of feature preservation vs adaptation
train_video3d_gradual_unfreeze() {
    echo ""
    echo "============================================================================="
    echo "EXPERIMENT 5: Video ResNet-3D with Gradual Unfreezing"
    echo "============================================================================="
    echo "Phase 1: Train classifier only (20 epochs)"
    echo "Phase 2: Unfreeze and fine-tune all (80 epochs)"
    echo ""

    # Phase 1: Frozen backbone
    python main.py \
      --model_name resnet18_video3d \
      --dataset_name nlst \
      --train \
      --monitor_key val_auc \
      \
      --resnet18_video3d.pretrained true \
      --resnet18_video3d.freeze_backbone true \
      --resnet18_video3d.num_classes 2 \
      --resnet18_video3d.init_lr 1e-3 \
      --resnet18_video3d.use_attention_pooling true \
      --resnet18_video3d.use_localization_reg false \
      \
      --nlst.data_percent 100 \
      --nlst.use_data_augmentation true \
      --nlst.class_balance true \
      --nlst.batch_size ${BATCH_VIDEO3D} \
      --nlst.num_workers ${NUM_WORKERS} \
      --nlst.img_size "${IMG_SIZE}" \
      --nlst.num_images ${NUM_SLICES} \
      \
      --trainer.max_epochs 20 \
      --trainer.precision ${PRECISION} \
      --trainer.gradient_clip_val ${GRADIENT_CLIP} \
      --trainer.devices ${DEVICES} \
      --trainer.strategy ddp \
      --trainer.log_every_n_steps 5 \
      \
      --project_name nlst_b200_video3d_gradual_phase1

    # Find best checkpoint from phase 1
    PHASE1_CKPT=$(find models/nlst_b200_video3d_gradual_phase1 -name "*.ckpt" ! -name "last.ckpt" | head -1)

    if [ -n "$PHASE1_CKPT" ]; then
        echo "Phase 2: Continuing from ${PHASE1_CKPT} with unfrozen backbone"

        # Phase 2: Unfreeze and continue training
        python main.py \
          --model_name resnet18_video3d \
          --dataset_name nlst \
          --train \
          --monitor_key val_auc \
          --ckpt_path "${PHASE1_CKPT}" \
          \
          --resnet18_video3d.pretrained true \
          --resnet18_video3d.freeze_backbone false \
          --resnet18_video3d.num_classes 2 \
          --resnet18_video3d.init_lr 3e-5 \
          --resnet18_video3d.use_attention_pooling true \
          --resnet18_video3d.use_localization_reg true \
          --resnet18_video3d.localization_reg_weight 0.1 \
          \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size ${BATCH_VIDEO3D} \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          \
          --trainer.max_epochs 80 \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --trainer.log_every_n_steps 5 \
          \
          --project_name nlst_b200_video3d_gradual_phase2
    else
        echo "WARNING: Could not find phase 1 checkpoint, skipping phase 2"
    fi

    echo "Gradual unfreezing experiment complete!"
}

# =============================================================================
# EXPERIMENT 6: Low Learning Rate from Scratch
# =============================================================================
# Tests if pretrained models work better simply because they use lower LR
# If from-scratch with low LR matches pretrained, it's optimization preconditioning
train_video3d_scratch_low_lr() {
    echo ""
    echo "============================================================================="
    echo "EXPERIMENT 6: Video ResNet-3D From Scratch with LOW Learning Rate"
    echo "============================================================================="
    echo "Uses same learning rate as pretrained model (3e-4 instead of 1e-3)"
    echo "Tests if pretraining benefit is just from allowing lower LR"
    echo ""

    python main.py \
      --model_name resnet18_video3d \
      --dataset_name nlst \
      --train \
      --monitor_key val_auc \
      \
      --resnet18_video3d.pretrained false \
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
      --project_name nlst_b200_video3d_scratch_low_lr

    echo "From scratch with low LR complete!"
}

# =============================================================================
# EXPERIMENT 7: Longer Training from Scratch
# =============================================================================
# Tests if from-scratch just needs more epochs to match pretrained
train_video3d_scratch_long() {
    echo ""
    echo "============================================================================="
    echo "EXPERIMENT 7: Video ResNet-3D From Scratch with EXTENDED Training"
    echo "============================================================================="
    echo "Trains for 200 epochs (2× normal)"
    echo "Tests if from-scratch just needs more training time"
    echo ""

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
      --resnet18_video3d.use_attention_pooling true \
      --resnet18_video3d.use_localization_reg true \
      --resnet18_video3d.localization_reg_weight 0.1 \
      \
      --nlst.data_percent 100 \
      --nlst.use_data_augmentation true \
      --nlst.class_balance true \
      --nlst.batch_size ${BATCH_VIDEO3D} \
      --nlst.num_workers ${NUM_WORKERS} \
      --nlst.img_size "${IMG_SIZE}" \
      --nlst.num_images ${NUM_SLICES} \
      \
      --trainer.max_epochs 200 \
      --trainer.precision ${PRECISION} \
      --trainer.gradient_clip_val ${GRADIENT_CLIP} \
      --trainer.devices ${DEVICES} \
      --trainer.strategy ddp \
      --trainer.log_every_n_steps 5 \
      \
      --project_name nlst_b200_video3d_scratch_extended

    echo "Extended training from scratch complete!"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo ""
echo "============================================================================="
echo "STARTING ALL EXPERIMENTS"
echo "============================================================================="
echo ""
echo "Order of execution:"
echo "  1. CNN3D baseline (no pretraining)"
echo "  2. ResNet-18 3D with ImageNet pretraining"
echo "  3. ResNet-18 3D without pretraining"
echo "  4. Video ResNet-3D with Kinetics pretraining"
echo "  5. Video ResNet-3D without pretraining"
echo "  6. Frozen backbone experiments (feature transfer test)"
echo "  7. Ablation experiments (optimization preconditioning test)"
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."

# Core experiments (required for report)
train_cnn3d
train_resnet3d_pretrained
train_resnet3d_scratch
train_video3d_pretrained
train_video3d_scratch

# Feature transfer experiments
train_video3d_frozen
train_resnet3d_frozen

# Optimization preconditioning experiments
train_video3d_scratch_low_lr
train_video3d_scratch_long
train_video3d_gradual_unfreeze

echo ""
echo "============================================================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "============================================================================="
echo ""
echo "Results saved in models/ directory:"
echo "  - nlst_b200_cnn3d_baseline"
echo "  - nlst_b200_resnet3d_imagenet_pretrained"
echo "  - nlst_b200_resnet3d_from_scratch"
echo "  - nlst_b200_video3d_kinetics_pretrained"
echo "  - nlst_b200_video3d_from_scratch"
echo "  - nlst_b200_video3d_frozen_backbone"
echo "  - nlst_b200_resnet3d_frozen_backbone"
echo "  - nlst_b200_video3d_scratch_low_lr"
echo "  - nlst_b200_video3d_scratch_extended"
echo "  - nlst_b200_video3d_gradual_phase1 / phase2"
echo ""
echo "Use compare_checkpoints.py to analyze all models and generate comparison plots."
echo ""
