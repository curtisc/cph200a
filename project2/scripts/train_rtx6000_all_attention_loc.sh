#!/bin/bash

################################################################################
# SUPER SCRIPT: Run All Models with Attention + Localization (RTX 6000 Pro)
################################################################################
# Runs 3 experiments in serial:
#   1. CNN3D with attention + localization
#   2. ResNet18_3D with attention + localization
#   3. ResNet18Video3D with attention + localization
#
# Total estimated time: ~72-96 hours (24-32 hours per model)
# System: 2× RTX 6000 Pro GPUs (96GB), 128GB RAM, PCIe x8 (no NVLink)
################################################################################

set -e  # Exit on error

echo "=============================================================================="
echo "STARTING ALL ATTENTION + LOCALIZATION EXPERIMENTS (RTX 6000 Pro)"
echo "=============================================================================="
echo "Hardware: 2× RTX 6000 Pro (96GB), 128GB RAM, 16-core CPU"
echo "This will run 3 models in serial:"
echo "  1. CNN3D (24-32 hours)"
echo "  2. ResNet18_3D (32-40 hours)"
echo "  3. ResNet18Video3D (32-40 hours)"
echo ""
echo "Total estimated time: 88-112 hours (~4 days)"
echo "Press Ctrl+C within 10 seconds to cancel..."
echo "=============================================================================="
sleep 10

# Track start time
SCRIPT_START=$(date +%s)

################################################################################
# EXPERIMENT 1: CNN3D with Attention + Localization
################################################################################
echo ""
echo "=============================================================================="
echo "EXPERIMENT 1/3: CNN3D with Attention + Localization"
echo "=============================================================================="
echo "Started at: $(date)"
echo ""

EXP1_START=$(date +%s)
bash /home/curtisc/cph200a/project2/scripts/train_rtx6000_cnn3d_attention_loc.sh
EXP1_END=$(date +%s)
EXP1_DURATION=$((EXP1_END - EXP1_START))

echo ""
echo "EXPERIMENT 1 COMPLETE!"
echo "CNN3D Duration: $((EXP1_DURATION / 3600))h $((EXP1_DURATION % 3600 / 60))m"
echo "Finished at: $(date)"
echo ""

################################################################################
# EXPERIMENT 2: ResNet18_3D with Attention + Localization
################################################################################
echo ""
echo "=============================================================================="
echo "EXPERIMENT 2/3: ResNet18_3D with Attention + Localization"
echo "=============================================================================="
echo "Started at: $(date)"
echo ""

EXP2_START=$(date +%s)
bash /home/curtisc/cph200a/project2/scripts/train_rtx6000_resnet3d_attention_loc.sh
EXP2_END=$(date +%s)
EXP2_DURATION=$((EXP2_END - EXP2_START))

echo ""
echo "EXPERIMENT 2 COMPLETE!"
echo "ResNet18_3D Duration: $((EXP2_DURATION / 3600))h $((EXP2_DURATION % 3600 / 60))m"
echo "Finished at: $(date)"
echo ""

################################################################################
# EXPERIMENT 3: ResNet18Video3D with Attention + Localization
################################################################################
echo ""
echo "=============================================================================="
echo "EXPERIMENT 3/3: ResNet18Video3D with Attention + Localization"
echo "=============================================================================="
echo "Started at: $(date)"
echo ""

EXP3_START=$(date +%s)
bash /home/curtisc/cph200a/project2/scripts/train_rtx6000_video3d_attention_loc.sh
EXP3_END=$(date +%s)
EXP3_DURATION=$((EXP3_END - EXP3_START))

echo ""
echo "EXPERIMENT 3 COMPLETE!"
echo "ResNet18Video3D Duration: $((EXP3_DURATION / 3600))h $((EXP3_DURATION % 3600 / 60))m"
echo "Finished at: $(date)"
echo ""

################################################################################
# FINAL SUMMARY
################################################################################
SCRIPT_END=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END - SCRIPT_START))

echo ""
echo "=============================================================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "=============================================================================="
echo ""
echo "Summary:"
echo "  CNN3D:            $((EXP1_DURATION / 3600))h $((EXP1_DURATION % 3600 / 60))m"
echo "  ResNet18_3D:      $((EXP2_DURATION / 3600))h $((EXP2_DURATION % 3600 / 60))m"
echo "  ResNet18Video3D:  $((EXP3_DURATION / 3600))h $((EXP3_DURATION % 3600 / 60))m"
echo "  ─────────────────────────────────────"
echo "  Total:            $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m"
echo ""
echo "Finished at: $(date)"
echo ""
echo "Check WandB for results:"
echo "  - nlst_rtx6000_cnn3d_attention_loc"
echo "  - nlst_rtx6000_resnet3d_attention_loc"
echo "  - nlst_rtx6000_video3d_attention_loc"
echo ""
echo "Compare metrics:"
echo "  - val_auc (classification performance)"
echo "  - val_iou (localization quality)"
echo "  - val_dice (localization quality)"
echo ""
echo "Note: Training times are ~4x longer than H100 due to fewer GPUs."
echo "Gradient sync overhead is negligible (<1%) thanks to NCCL flags."
echo "=============================================================================="
