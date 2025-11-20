#!/usr/bin/env python3
"""
Clinical Analysis Script: Compare model predictions to LungRads and analyze impact

This script:
1. Loads a trained model checkpoint
2. Generates predictions on test set
3. Compares model to LungRads criteria
4. Performs subgroup analysis
5. Simulates clinical workflows
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from scipy import stats
import argparse
from tqdm import tqdm

# Lightning imports
import lightning.pytorch as pl
from src.lightning import ResNet18Video3D, ResNet18_3D, CNN3D
from src.dataset import NLST


def load_model_and_data(checkpoint_path, config):
    """Load trained model and data module"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Determine model type from checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})

    # Load model
    model = ResNet18Video3D.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Load data module with config from checkpoint
    # Use checkpoint's original config, but allow overrides for batch_size/num_workers
    datamodule = NLST(
        batch_size=config.get('batch_size', 1),
        num_workers=config.get('num_workers', 4),
        img_size=hparams.get('img_size', [256, 256]),
        num_images=hparams.get('num_images', 200),
        use_data_augmentation=False,  # No augmentation during inference
        class_balance=False,  # No sampling during inference
    )
    datamodule.setup()

    return model, datamodule


def get_predictions(model, dataloader, device='cuda'):
    """Get model predictions and metadata for all samples"""
    model = model.to(device)
    model.eval()

    all_probs = []
    all_labels = []
    all_pids = []
    all_lung_rads = []
    all_time_at_event = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating predictions")):
            x = batch['x'].to(device)
            y = batch['y_seq'][:, 0]  # 1-year prediction

            # Forward pass
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # Probability of cancer

            batch_size = len(y)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Handle potentially missing metadata fields
            if 'pid' in batch and batch['pid'] is not None:
                pids = batch['pid'] if isinstance(batch['pid'], list) else [batch['pid']]
                all_pids.extend(pids)
            else:
                all_pids.extend([f'patient_{batch_idx}_{i}' for i in range(batch_size)])

            if 'lung_rads' in batch and batch['lung_rads'] is not None:
                lr = batch['lung_rads']
                if torch.is_tensor(lr):
                    all_lung_rads.extend(lr.cpu().numpy())
                else:
                    all_lung_rads.extend(lr if isinstance(lr, list) else [lr])
            else:
                all_lung_rads.extend([None] * batch_size)

            if 'time_at_event' in batch and batch['time_at_event'] is not None:
                tae = batch['time_at_event']
                if torch.is_tensor(tae):
                    all_time_at_event.extend(tae.cpu().numpy())
                else:
                    all_time_at_event.extend(tae if isinstance(tae, list) else [tae])
            else:
                all_time_at_event.extend([None] * batch_size)

    df = pd.DataFrame({
        'pid': all_pids,
        'y_true': all_labels,
        'model_prob': all_probs,
        'lung_rads': all_lung_rads,
        'time_at_event': all_time_at_event,
    })

    # Check if we have lung_rads data
    if df['lung_rads'].isna().all():
        print("\nWARNING: No LungRads data found in batches!")
        print("LungRads comparison will be skipped.")
        print("Make sure 'lung_rads' is included in your dataset's __getitem__ method.")

    return df


def compare_to_lungrads(df):
    """Compare model performance to LungRads criteria"""
    print("\n" + "="*80)
    print("MODEL VS LUNGRADS COMPARISON")
    print("="*80)

    # LungRads performance
    lr_auc = roc_auc_score(df['y_true'], df['lung_rads'])

    # Model performance
    model_auc = roc_auc_score(df['y_true'], df['model_prob'])

    print(f"\nLungRads AUC: {lr_auc:.4f}")
    print(f"Model AUC:    {model_auc:.4f}")
    print(f"Improvement:  {model_auc - lr_auc:.4f} ({(model_auc - lr_auc)/lr_auc*100:.1f}%)")

    # Sensitivity/Specificity at different operating points
    print("\n" + "-"*80)
    print("OPERATING CHARACTERISTICS")
    print("-"*80)

    # LungRads operating point (threshold at 0.5 for binary)
    lr_preds = (df['lung_rads'] > 0.5).astype(int)
    lr_cm = confusion_matrix(df['y_true'], lr_preds)
    lr_sens = lr_cm[1, 1] / (lr_cm[1, 1] + lr_cm[1, 0])
    lr_spec = lr_cm[0, 0] / (lr_cm[0, 0] + lr_cm[0, 1])
    lr_ppv = lr_cm[1, 1] / (lr_cm[1, 1] + lr_cm[0, 1])
    lr_npv = lr_cm[0, 0] / (lr_cm[0, 0] + lr_cm[1, 0])

    print(f"\nLungRads (threshold=0.5):")
    print(f"  Sensitivity: {lr_sens:.3f}")
    print(f"  Specificity: {lr_spec:.3f}")
    print(f"  PPV:         {lr_ppv:.3f}")
    print(f"  NPV:         {lr_npv:.3f}")

    # Model at matched sensitivity
    fpr, tpr, thresholds = roc_curve(df['y_true'], df['model_prob'])
    # Find threshold that matches LungRads sensitivity
    idx = np.argmin(np.abs(tpr - lr_sens))
    model_thresh_matched = thresholds[idx]
    model_preds_matched = (df['model_prob'] > model_thresh_matched).astype(int)
    model_cm_matched = confusion_matrix(df['y_true'], model_preds_matched)
    model_sens_matched = model_cm_matched[1, 1] / (model_cm_matched[1, 1] + model_cm_matched[1, 0])
    model_spec_matched = model_cm_matched[0, 0] / (model_cm_matched[0, 0] + model_cm_matched[0, 1])
    model_ppv_matched = model_cm_matched[1, 1] / (model_cm_matched[1, 1] + model_cm_matched[0, 1])

    print(f"\nModel (matched sensitivity={model_sens_matched:.3f}):")
    print(f"  Threshold:   {model_thresh_matched:.3f}")
    print(f"  Specificity: {model_spec_matched:.3f} (vs {lr_spec:.3f} LungRads)")
    print(f"  PPV:         {model_ppv_matched:.3f} (vs {lr_ppv:.3f} LungRads)")
    print(f"  Spec gain:   {model_spec_matched - lr_spec:.3f} ({(model_spec_matched - lr_spec)/lr_spec*100:.1f}%)")

    return {
        'lungrads_auc': lr_auc,
        'model_auc': model_auc,
        'auc_improvement': model_auc - lr_auc,
        'lungrads_sensitivity': lr_sens,
        'lungrads_specificity': lr_spec,
        'lungrads_ppv': lr_ppv,
        'model_sensitivity_matched': model_sens_matched,
        'model_specificity_matched': model_spec_matched,
        'model_ppv_matched': model_ppv_matched,
        'specificity_gain': model_spec_matched - lr_spec,
        'threshold_matched_sens': model_thresh_matched,
    }


def simulate_clinical_workflow(df, metrics):
    """Simulate impact of model-assisted workflow"""
    print("\n" + "="*80)
    print("CLINICAL WORKFLOW SIMULATION")
    print("="*80)

    total_patients = len(df)
    total_cancer = df['y_true'].sum()

    print(f"\nCohort: {total_patients} patients, {total_cancer} with cancer ({total_cancer/total_patients*100:.1f}%)")

    # Scenario 1: LungRads only (current standard)
    lr_positives = (df['lung_rads'] > 0.5).sum()
    lr_true_positives = ((df['lung_rads'] > 0.5) & (df['y_true'] == 1)).sum()
    lr_false_positives = ((df['lung_rads'] > 0.5) & (df['y_true'] == 0)).sum()

    print("\n" + "-"*80)
    print("Scenario 1: LungRads Only (Current Standard)")
    print("-"*80)
    print(f"  Positive screens: {lr_positives} ({lr_positives/total_patients*100:.1f}%)")
    print(f"  True positives:   {lr_true_positives}")
    print(f"  False positives:  {lr_false_positives}")
    print(f"  Cancers detected: {lr_true_positives}/{total_cancer} ({lr_true_positives/total_cancer*100:.1f}%)")

    # Scenario 2: Model at matched sensitivity
    threshold = metrics['threshold_matched_sens']
    model_positives = (df['model_prob'] > threshold).sum()
    model_true_positives = ((df['model_prob'] > threshold) & (df['y_true'] == 1)).sum()
    model_false_positives = ((df['model_prob'] > threshold) & (df['y_true'] == 0)).sum()

    print("\n" + "-"*80)
    print(f"Scenario 2: AI Model (threshold={threshold:.3f}, matched sensitivity)")
    print("-"*80)
    print(f"  Positive screens: {model_positives} ({model_positives/total_patients*100:.1f}%)")
    print(f"  True positives:   {model_true_positives}")
    print(f"  False positives:  {model_false_positives}")
    print(f"  Cancers detected: {model_true_positives}/{total_cancer} ({model_true_positives/total_cancer*100:.1f}%)")

    fp_reduction = lr_false_positives - model_false_positives
    print(f"\n  False positive reduction: {fp_reduction} ({fp_reduction/lr_false_positives*100:.1f}%)")
    print(f"  Unnecessary follow-ups avoided: {fp_reduction}")

    # Scenario 3: Hybrid workflow - AI triage
    print("\n" + "-"*80)
    print("Scenario 3: AI-Assisted Triage (Proposed Workflow)")
    print("-"*80)
    print("\nWorkflow:")
    print("  1. All patients screened with CT")
    print("  2. AI model evaluates all scans")
    print("  3. High-risk (AI > threshold): immediate follow-up")
    print("  4. Low-risk (AI <= threshold): defer or extend interval")
    print()

    # Triage into risk groups
    high_risk = df['model_prob'] > threshold
    low_risk = ~high_risk

    high_risk_cancer = (high_risk & (df['y_true'] == 1)).sum()
    low_risk_cancer = (low_risk & (df['y_true'] == 1)).sum()

    print(f"  High-risk group: {high_risk.sum()} patients ({high_risk.sum()/total_patients*100:.1f}%)")
    print(f"    - Cancers: {high_risk_cancer} ({high_risk_cancer/high_risk.sum()*100:.2f}% prevalence)")
    print(f"  Low-risk group:  {low_risk.sum()} patients ({low_risk.sum()/total_patients*100:.1f}%)")
    print(f"    - Cancers: {low_risk_cancer} ({low_risk_cancer/low_risk.sum()*100:.2f}% prevalence)")

    return {
        'total_patients': total_patients,
        'total_cancer': total_cancer,
        'lr_false_positives': lr_false_positives,
        'model_false_positives': model_false_positives,
        'fp_reduction': fp_reduction,
        'fp_reduction_pct': fp_reduction/lr_false_positives*100,
        'high_risk_count': high_risk.sum(),
        'high_risk_cancer_prevalence': high_risk_cancer/high_risk.sum()*100,
        'low_risk_count': low_risk.sum(),
        'low_risk_cancer_prevalence': low_risk_cancer/low_risk.sum()*100,
    }


def subgroup_analysis(df, metrics):
    """Analyze model performance across subgroups

    Note: This function template assumes you have demographic data available.
    You'll need to add demographic fields to the dataloader/dataset.
    """
    print("\n" + "="*80)
    print("SUBGROUP ANALYSIS")
    print("="*80)

    print("\nNote: Subgroup analysis requires demographic data (age, sex, race, etc.)")
    print("Add these fields to your dataset.py and dataloader for full analysis.")

    # Example: Age-based analysis (if available)
    # You would need to add 'age' to the batch in dataset.py
    if 'age' in df.columns:
        print("\n" + "-"*80)
        print("Performance by Age Group")
        print("-"*80)

        # Create age bins
        df['age_group'] = pd.cut(df['age'], bins=[0, 60, 70, 100], labels=['<60', '60-70', '>70'])

        for age_group in df['age_group'].unique():
            if pd.isna(age_group):
                continue
            subset = df[df['age_group'] == age_group]
            auc = roc_auc_score(subset['y_true'], subset['model_prob'])
            print(f"  {age_group}: AUC={auc:.3f} (n={len(subset)})")

    # Template for other subgroups
    # TODO: Add sex, race, smoking history, etc.

    return {}


def plot_results(df, metrics, output_dir='./clinical_analysis_plots'):
    """Generate visualization plots"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. ROC curves comparison
    plt.figure(figsize=(10, 8))

    # LungRads ROC
    fpr_lr, tpr_lr, _ = roc_curve(df['y_true'], df['lung_rads'])
    plt.plot(fpr_lr, tpr_lr, label=f"LungRads (AUC={metrics['lungrads_auc']:.3f})", linewidth=2)

    # Model ROC
    fpr_model, tpr_model, _ = roc_curve(df['y_true'], df['model_prob'])
    plt.plot(fpr_model, tpr_model, label=f"AI Model (AUC={metrics['model_auc']:.3f})", linewidth=2)

    # Diagonal
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve: Model vs LungRads', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_comparison.png', dpi=300)
    plt.close()

    # 2. Calibration plot
    plt.figure(figsize=(10, 8))

    # Bin predictions
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    model_calibration = []
    for i in range(len(bins)-1):
        mask = (df['model_prob'] >= bins[i]) & (df['model_prob'] < bins[i+1])
        if mask.sum() > 0:
            model_calibration.append(df[mask]['y_true'].mean())
        else:
            model_calibration.append(np.nan)

    plt.plot(bin_centers, model_calibration, 'o-', label='Model', markersize=8, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')

    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Observed Frequency', fontsize=12)
    plt.title('Calibration Plot', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/calibration.png', dpi=300)
    plt.close()

    # 3. Risk distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Cancer cases
    axes[0].hist(df[df['y_true'] == 1]['model_prob'], bins=30, alpha=0.7, label='Cancer', color='red')
    axes[0].axvline(metrics['threshold_matched_sens'], color='black', linestyle='--',
                   label=f'Threshold={metrics["threshold_matched_sens"]:.3f}')
    axes[0].set_xlabel('Model Probability', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Risk Distribution: Cancer Cases', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Non-cancer cases
    axes[1].hist(df[df['y_true'] == 0]['model_prob'], bins=30, alpha=0.7, label='No Cancer', color='blue')
    axes[1].axvline(metrics['threshold_matched_sens'], color='black', linestyle='--',
                   label=f'Threshold={metrics["threshold_matched_sens"]:.3f}')
    axes[1].set_xlabel('Model Probability', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Risk Distribution: Non-Cancer Cases', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/risk_distributions.png', dpi=300)
    plt.close()

    print(f"\nPlots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Clinical analysis of lung cancer screening model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='./clinical_analysis_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and data
    config = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
    }
    model, datamodule = load_model_and_data(args.checkpoint, config)

    # Get predictions on test set
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS ON TEST SET")
    print("="*80)
    test_loader = datamodule.test_dataloader()
    df = get_predictions(model, test_loader, device=args.device)

    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to: {predictions_path}")

    # Compare to LungRads
    metrics = compare_to_lungrads(df)

    # Simulate clinical workflows
    workflow_metrics = simulate_clinical_workflow(df, metrics)

    # Subgroup analysis
    subgroup_metrics = subgroup_analysis(df, metrics)

    # Generate plots
    plot_dir = os.path.join(args.output_dir, 'plots')
    plot_results(df, metrics, output_dir=plot_dir)

    # Save metrics
    all_metrics = {**metrics, **workflow_metrics, **subgroup_metrics}
    metrics_df = pd.DataFrame([all_metrics])
    metrics_path = os.path.join(args.output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
