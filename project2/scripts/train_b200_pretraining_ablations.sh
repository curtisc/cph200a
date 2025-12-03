#!/bin/bash

# =============================================================================
# PRETRAINING ABLATION EXPERIMENTS - B200 Configuration
# =============================================================================
# Experiments to understand WHY pretraining helps:
# - Feature Transfer: Are the learned features directly useful?
# - Optimization Preconditioning: Is it just about better initialization?
#
# Experiments:
# 1. Frozen Backbone (tests pure feature transfer)
# 2. Low LR from Scratch (tests if pretrain just enables lower LR)
# 3. Extended Training (tests if scratch just needs more epochs)
# 4. Gradual Unfreezing (tests optimal balance)
# =============================================================================

set -e

# B200 Configuration
DEVICES=8
PRECISION="bf16-mixed"
NUM_WORKERS=24
GRADIENT_CLIP=1.0

# Native resolution for B200 (raw data is 256×256)
IMG_SIZE="[256, 256]"
NUM_SLICES=250
BATCH_SIZE=10

# Data paths (adjust DATA_ROOT as needed for your system)
DATA_ROOT=~/cphdata
NLST_DIR="${DATA_ROOT}/compressed"
NLST_METADATA="${DATA_ROOT}/nlst-metadata/full_nlst_google.json"
VALID_EXAM_PATH="${DATA_ROOT}/nlst-metadata/valid_exams.p"
LUNGRADS_PATH="${DATA_ROOT}/nlst-metadata/nlst_acc2lungrads.p"

echo "============================================================================="
echo "B200 PRETRAINING ABLATION EXPERIMENTS"
echo "============================================================================="
echo "Understanding WHY pretraining helps model performance"
echo ""

# Select which experiment to run
EXPERIMENT=${1:-"all"}

case $EXPERIMENT in
    "frozen_video3d")
        echo "EXPERIMENT: Video ResNet-3D with FROZEN Backbone"
        echo "Tests pure feature transfer - only classifier is trained"
        echo ""
        python main.py \
          --model_name resnet18_video3d \
          --dataset_name nlst \
          --train \
          --monitor_key val_auc \
          --resnet18_video3d.pretrained true \
          --resnet18_video3d.freeze_backbone true \
          --resnet18_video3d.num_classes 2 \
          --resnet18_video3d.init_lr 1e-3 \
          --resnet18_video3d.use_attention_pooling true \
          --nlst.nlst_dir "${NLST_DIR}" \
          --nlst.nlst_metadata_path "${NLST_METADATA}" \
          --nlst.valid_exam_path "${VALID_EXAM_PATH}" \
          --nlst.lungrads_path "${LUNGRADS_PATH}" \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size ${BATCH_SIZE} \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          --trainer.max_epochs 100 \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --project_name nlst_b200_video3d_frozen_backbone
        ;;

    "frozen_resnet3d")
        echo "EXPERIMENT: ResNet-18 3D with FROZEN Backbone"
        echo "Tests pure feature transfer from ImageNet"
        echo ""
        python main.py \
          --model_name resnet18_3d \
          --dataset_name nlst \
          --train \
          --monitor_key val_auc \
          --resnet18_3d.pretrained true \
          --resnet18_3d.freeze_backbone true \
          --resnet18_3d.num_classes 2 \
          --resnet18_3d.init_lr 1e-3 \
          --resnet18_3d.use_attention_pooling true \
          --nlst.nlst_dir "${NLST_DIR}" \
          --nlst.nlst_metadata_path "${NLST_METADATA}" \
          --nlst.valid_exam_path "${VALID_EXAM_PATH}" \
          --nlst.lungrads_path "${LUNGRADS_PATH}" \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size 8 \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          --trainer.max_epochs 100 \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --project_name nlst_b200_resnet3d_frozen_backbone
        ;;

    "scratch_low_lr")
        echo "EXPERIMENT: From Scratch with LOW Learning Rate"
        echo "Tests if pretraining benefit is just from enabling lower LR"
        echo "Uses same LR as pretrained (3e-4) instead of typical 1e-3"
        echo ""
        python main.py \
          --model_name resnet18_video3d \
          --dataset_name nlst \
          --train \
          --monitor_key val_auc \
          --resnet18_video3d.pretrained false \
          --resnet18_video3d.freeze_backbone false \
          --resnet18_video3d.num_classes 2 \
          --resnet18_video3d.init_lr 3e-4 \
          --resnet18_video3d.use_attention_pooling true \
          --resnet18_video3d.use_localization_reg true \
          --resnet18_video3d.localization_reg_weight 0.1 \
          --nlst.nlst_dir "${NLST_DIR}" \
          --nlst.nlst_metadata_path "${NLST_METADATA}" \
          --nlst.valid_exam_path "${VALID_EXAM_PATH}" \
          --nlst.lungrads_path "${LUNGRADS_PATH}" \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size ${BATCH_SIZE} \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          --trainer.max_epochs 100 \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --project_name nlst_b200_video3d_scratch_low_lr
        ;;

    "scratch_extended")
        echo "EXPERIMENT: From Scratch with EXTENDED Training"
        echo "Tests if from-scratch just needs more epochs"
        echo "Training for 200 epochs (2× normal)"
        echo ""
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
          --nlst.nlst_dir "${NLST_DIR}" \
          --nlst.nlst_metadata_path "${NLST_METADATA}" \
          --nlst.valid_exam_path "${VALID_EXAM_PATH}" \
          --nlst.lungrads_path "${LUNGRADS_PATH}" \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size ${BATCH_SIZE} \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          --trainer.max_epochs 200 \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --project_name nlst_b200_video3d_scratch_extended
        ;;

    "scratch_very_extended")
        echo "EXPERIMENT: From Scratch with VERY EXTENDED Training"
        echo "Tests if from-scratch can eventually match pretrained"
        echo "Training for 300 epochs (3× normal)"
        echo ""
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
          --nlst.nlst_dir "${NLST_DIR}" \
          --nlst.nlst_metadata_path "${NLST_METADATA}" \
          --nlst.valid_exam_path "${VALID_EXAM_PATH}" \
          --nlst.lungrads_path "${LUNGRADS_PATH}" \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size ${BATCH_SIZE} \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          --trainer.max_epochs 300 \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --project_name nlst_b200_video3d_scratch_very_extended
        ;;

    "warmup_scratch")
        echo "EXPERIMENT: From Scratch with Learning Rate Warmup"
        echo "Tests if scratch training benefits from warmup like pretrained"
        echo ""
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
          --nlst.nlst_dir "${NLST_DIR}" \
          --nlst.nlst_metadata_path "${NLST_METADATA}" \
          --nlst.valid_exam_path "${VALID_EXAM_PATH}" \
          --nlst.lungrads_path "${LUNGRADS_PATH}" \
          --nlst.data_percent 100 \
          --nlst.use_data_augmentation true \
          --nlst.class_balance true \
          --nlst.batch_size ${BATCH_SIZE} \
          --nlst.num_workers ${NUM_WORKERS} \
          --nlst.img_size "${IMG_SIZE}" \
          --nlst.num_images ${NUM_SLICES} \
          --trainer.max_epochs 100 \
          --trainer.precision ${PRECISION} \
          --trainer.gradient_clip_val ${GRADIENT_CLIP} \
          --trainer.devices ${DEVICES} \
          --trainer.strategy ddp \
          --project_name nlst_b200_video3d_scratch_warmup
        ;;

    "partial_freeze")
        echo "EXPERIMENT: Partial Freeze (only early layers frozen)"
        echo "Tests which layers benefit most from pretraining"
        echo ""
        # This would require model modification to freeze specific layers
        echo "NOTE: This experiment requires model code changes to freeze specific layers"
        echo "Skipping for now - implement by modifying lightning.py if needed"
        ;;

    "all")
        echo "Running ALL ablation experiments..."
        $0 frozen_video3d
        $0 frozen_resnet3d
        $0 scratch_low_lr
        $0 scratch_extended
        echo ""
        echo "All ablation experiments complete!"
        ;;

    *)
        echo "Usage: $0 {frozen_video3d|frozen_resnet3d|scratch_low_lr|scratch_extended|scratch_very_extended|warmup_scratch|all}"
        echo ""
        echo "Experiments:"
        echo "  frozen_video3d      - Video3D with frozen backbone (pure feature transfer)"
        echo "  frozen_resnet3d     - ResNet3D with frozen backbone (pure feature transfer)"
        echo "  scratch_low_lr      - From scratch with low LR (tests if benefit is just LR)"
        echo "  scratch_extended    - From scratch 200 epochs (tests if just needs more time)"
        echo "  scratch_very_extended - From scratch 300 epochs"
        echo "  warmup_scratch      - From scratch with warmup"
        echo "  all                 - Run core ablation experiments"
        exit 1
        ;;
esac

echo ""
echo "Experiment complete!"
