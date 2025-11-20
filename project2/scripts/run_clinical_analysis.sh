#!/bin/bash
#
# Complete Clinical Analysis Pipeline
#
# This script runs all three phases of the clinical analysis:
# 1. Extract demographics
# 2. Generate predictions and compare to LungRads
# 3. Perform subgroup analysis
#

set -e  # Exit on error

echo "================================================================================================="
echo "                       CLINICAL ANALYSIS PIPELINE                                                "
echo "================================================================================================="
echo ""

# Configuration
CHECKPOINT="../models/nlst_h100_video3d_attention_loc-2/9n6es6fl/checkpoints/last.ckpt"
METADATA_PATH="../../../data/project2/nlst-metadata/full_nlst_google.json"
OUTPUT_DIR="./clinical_analysis_results"
BATCH_SIZE=1
NUM_WORKERS=4

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Num workers: $NUM_WORKERS"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    find models/ -name "*.ckpt" | head -10
    echo ""
    echo "Please update the CHECKPOINT variable in this script."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Extract Demographics
echo ""
echo "================================================================================================="
echo "STEP 1: Extracting Demographics"
echo "================================================================================================="
echo ""

if [ ! -f "$OUTPUT_DIR/demographics.csv" ]; then
    python extract_demographics.py \
        --metadata_path "$METADATA_PATH" \
        --output "$OUTPUT_DIR/demographics.csv"

    if [ $? -ne 0 ]; then
        echo "WARNING: Demographics extraction failed. Continuing without demographics..."
        echo "Subgroup analysis will be limited."
    fi
else
    echo "Demographics already extracted: $OUTPUT_DIR/demographics.csv"
    echo "Delete this file to re-extract."
fi

# Step 2: Generate Predictions and Compare to LungRads
echo ""
echo "================================================================================================="
echo "STEP 2: Generating Predictions and Comparing to LungRads"
echo "================================================================================================="
echo ""

python clinical_analysis.py \
    --checkpoint "$CHECKPOINT" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device cuda \
    --output_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Prediction generation failed!"
    exit 1
fi

# Step 3: Subgroup Analysis (if demographics available)
echo ""
echo "================================================================================================="
echo "STEP 3: Subgroup Analysis"
echo "================================================================================================="
echo ""

if [ -f "$OUTPUT_DIR/demographics.csv" ] && [ -f "$OUTPUT_DIR/predictions.csv" ]; then
    python subgroup_analysis.py \
        --predictions "$OUTPUT_DIR/predictions.csv" \
        --demographics "$OUTPUT_DIR/demographics.csv" \
        --output_dir "$OUTPUT_DIR/subgroup_analysis"

    if [ $? -ne 0 ]; then
        echo "WARNING: Subgroup analysis failed."
    fi
else
    echo "Skipping subgroup analysis (missing demographics or predictions)"
fi

# Summary
echo ""
echo "================================================================================================="
echo "                       ANALYSIS COMPLETE!                                                        "
echo "================================================================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
echo "  - predictions.csv          : Model predictions + LungRads scores"
echo "  - metrics.csv              : Summary performance metrics"
echo "  - plots/                   : Visualization plots"
echo "    - roc_comparison.png     : ROC curves (Model vs LungRads)"
echo "    - calibration.png        : Calibration plot"
echo "    - risk_distributions.png : Risk score distributions"

if [ -f "$OUTPUT_DIR/subgroup_analysis/subgroup_metrics.csv" ]; then
    echo "  - subgroup_analysis/       : Subgroup analysis results"
    echo "    - subgroup_metrics.csv   : Performance by demographic groups"
    echo "    - {subgroup}_performance.png : Plots for each subgroup"
fi

echo ""
echo "Next steps:"
echo "  1. Review the metrics in: $OUTPUT_DIR/metrics.csv"
echo "  2. Examine plots in: $OUTPUT_DIR/plots/"
echo "  3. Fill in CLINICAL_ANALYSIS_REPORT_TEMPLATE.md with your results"
echo ""
echo "For detailed guidance, see: CLINICAL_ANALYSIS_README.md"
echo ""
