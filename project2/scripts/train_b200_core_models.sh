#!/bin/bash

# =============================================================================
# CORE MODELS FOR PROJECT REPORT - B200 Configuration
# =============================================================================
# Trains the 5 required models for the project report:
# 1. Simple 3D CNN (no pretraining)
# 2. ResNet-18 3D with ImageNet pretraining
# 3. ResNet-18 3D without pretraining
# 4. Video ResNet-3D with Kinetics pretraining
# 5. Video ResNet-3D without pretraining
# =============================================================================

set -e

# B200 Configuration
DEVICES=8
PRECISION="bf16-mixed"
NUM_WORKERS=24
MAX_EPOCHS=100
GRADIENT_CLIP=1.0

# Ultra-high resolution for B200 (180GB VRAM)
IMG_SIZE="[512, 512]"
NUM_SLICES=256

echo "============================================================================="
echo "B200 CORE MODEL TRAINING"
echo "============================================================================="
echo "Training 5 core models for project report comparison"
echo "Resolution: 512×512×256 slices"
echo ""

# Select which model to train
MODEL=${1:-"all"}

case $MODEL in
    "cnn3d")
        echo "Training: Simple 3D CNN"
        python main.py \
          --model_name cnn3d \
          --dataset_name nlst \
          --train \
          --monitor_key val_auc \
          --cnn3d.input_channels 1 \
          --cnn3d.hidden_dim 256 \
          --cnn3d.num_layers 5 \
          --cnn3d.num_classes 2 \
          --cnn3d.use_bn true \
          --cnn3d.init_lr 3e-4 \
          --cnn3d.use_attention_pooling true \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size 6 \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          --trainer.max_epochs ${MAX_EPOCHS} \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --trainer.log_every_n_steps 5 \
          --project_name nlst_b200_cnn3d_baseline
        ;;

    "resnet3d_pretrained")
        echo "Training: ResNet-18 3D WITH ImageNet Pretraining"
        python main.py \
          --model_name resnet18_3d \
          --dataset_name nlst \
          --train \
          --monitor_key val_auc \
          --resnet18_3d.pretrained true \
          --resnet18_3d.freeze_backbone false \
          --resnet18_3d.num_classes 2 \
          --resnet18_3d.init_lr 3e-4 \
          --resnet18_3d.use_attention_pooling true \
          --resnet18_3d.use_localization_reg true \
          --resnet18_3d.localization_reg_weight 0.1 \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size 8 \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          --trainer.max_epochs ${MAX_EPOCHS} \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --trainer.log_every_n_steps 5 \
          --project_name nlst_b200_resnet3d_imagenet_pretrained
        ;;

    "resnet3d_scratch")
        echo "Training: ResNet-18 3D WITHOUT Pretraining"
        python main.py \
          --model_name resnet18_3d \
          --dataset_name nlst \
          --train \
          --monitor_key val_auc \
          --resnet18_3d.pretrained false \
          --resnet18_3d.freeze_backbone false \
          --resnet18_3d.num_classes 2 \
          --resnet18_3d.init_lr 1e-3 \
          --resnet18_3d.use_attention_pooling true \
          --resnet18_3d.use_localization_reg true \
          --resnet18_3d.localization_reg_weight 0.1 \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size 8 \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          --trainer.max_epochs ${MAX_EPOCHS} \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --trainer.log_every_n_steps 5 \
          --project_name nlst_b200_resnet3d_from_scratch
        ;;

    "video3d_pretrained")
        echo "Training: Video ResNet-3D WITH Kinetics Pretraining"
        python main.py \
          --model_name resnet18_video3d \
          --dataset_name nlst \
          --train \
          --monitor_key val_auc \
          --resnet18_video3d.pretrained true \
          --resnet18_video3d.freeze_backbone false \
          --resnet18_video3d.num_classes 2 \
          --resnet18_video3d.init_lr 3e-4 \
          --resnet18_video3d.use_attention_pooling true \
          --resnet18_video3d.use_localization_reg true \
          --resnet18_video3d.localization_reg_weight 0.1 \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size 6 \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          --trainer.max_epochs ${MAX_EPOCHS} \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --trainer.log_every_n_steps 5 \
          --project_name nlst_b200_video3d_kinetics_pretrained
        ;;

    "video3d_scratch")
        echo "Training: Video ResNet-3D WITHOUT Pretraining"
        python main.py \
          --model_name resnet18_video3d \
          --dataset_name nlst \
          --train \
          --monitor_key val_auc \
          --resnet18_video3d.pretrained false \
          --resnet18_video3d.freeze_backbone false \
          --resnet18_video3d.num_classes 2 \
          --resnet18_video3d.init_lr 1e-3 \
          --resnet18_video3d.use_attention_pooling true \
          --resnet18_video3d.use_localization_reg true \
          --resnet18_video3d.localization_reg_weight 0.1 \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size 6 \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          --trainer.max_epochs ${MAX_EPOCHS} \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --trainer.log_every_n_steps 5 \
          --project_name nlst_b200_video3d_from_scratch
        ;;

    "all")
        echo "Training ALL 5 core models sequentially..."
        $0 cnn3d
        $0 resnet3d_pretrained
        $0 resnet3d_scratch
        $0 video3d_pretrained
        $0 video3d_scratch
        echo ""
        echo "All 5 core models trained!"
        ;;

    *)
        echo "Usage: $0 {cnn3d|resnet3d_pretrained|resnet3d_scratch|video3d_pretrained|video3d_scratch|all}"
        echo ""
        echo "Models:"
        echo "  cnn3d              - Simple 3D CNN (no pretraining)"
        echo "  resnet3d_pretrained - ResNet-18 3D with ImageNet pretraining"
        echo "  resnet3d_scratch   - ResNet-18 3D without pretraining"
        echo "  video3d_pretrained - Video ResNet-3D with Kinetics pretraining"
        echo "  video3d_scratch    - Video ResNet-3D without pretraining"
        echo "  all                - Train all 5 models"
        exit 1
        ;;
esac

echo ""
echo "Training complete!"
