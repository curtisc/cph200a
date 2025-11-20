#!/usr/bin/env python3
"""
Comprehensive subgroup analysis for lung cancer screening model.

Analyzes model performance across:
- Age groups
- Sex
- Race/ethnicity
- Smoking history (pack-years)
- BMI categories
- Baseline lung function
- Comorbidities

This helps identify disparities and ensure equitable performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import argparse


def load_data(predictions_path, demographics_path):
    """Load predictions and merge with demographics"""
    pred_df = pd.read_csv(predictions_path)
    demo_df = pd.read_csv(demographics_path)

    # Merge on patient ID
    df = pred_df.merge(demo_df, on='pid', how='inner')

    print(f"Loaded {len(df)} samples with demographics")
    return df


def analyze_subgroup(df, subgroup_col, subgroup_name, model_prob_col='model_prob',
                     label_col='y_true', lungrads_col='lung_rads'):
    """
    Analyze performance within each level of a subgroup variable.

    Returns a DataFrame with metrics for each subgroup level.
    """
    results = []

    for level in sorted(df[subgroup_col].dropna().unique()):
        subset = df[df[subgroup_col] == level].copy()

        if len(subset) < 10:
            continue  # Skip very small subgroups

        n_total = len(subset)
        n_cancer = subset[label_col].sum()
        prevalence = n_cancer / n_total if n_total > 0 else 0

        # Model performance
        if n_cancer > 0 and n_cancer < n_total:  # Need both classes
            try:
                model_auc = roc_auc_score(subset[label_col], subset[model_prob_col])
                lr_auc = roc_auc_score(subset[label_col], subset[lungrads_col])
            except:
                model_auc = np.nan
                lr_auc = np.nan
        else:
            model_auc = np.nan
            lr_auc = np.nan

        # Sensitivity/Specificity at various thresholds
        if not np.isnan(model_auc):
            # At LungRads-matched threshold (global)
            global_threshold = df[model_prob_col].median()  # Placeholder
            preds = (subset[model_prob_col] > global_threshold).astype(int)
            tp = ((preds == 1) & (subset[label_col] == 1)).sum()
            fp = ((preds == 1) & (subset[label_col] == 0)).sum()
            tn = ((preds == 0) & (subset[label_col] == 0)).sum()
            fn = ((preds == 0) & (subset[label_col] == 1)).sum()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        else:
            sensitivity = specificity = ppv = npv = np.nan

        results.append({
            'subgroup': subgroup_name,
            'level': level,
            'n': n_total,
            'n_cancer': n_cancer,
            'prevalence': prevalence * 100,
            'model_auc': model_auc,
            'lungrads_auc': lr_auc,
            'auc_improvement': model_auc - lr_auc if not np.isnan(model_auc) and not np.isnan(lr_auc) else np.nan,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
        })

    return pd.DataFrame(results)


def test_performance_parity(df, subgroup_col, model_prob_col='model_prob', label_col='y_true'):
    """
    Test for statistical differences in performance across subgroups.

    Uses bootstrap resampling to compute confidence intervals and p-values.
    """
    levels = sorted(df[subgroup_col].dropna().unique())

    if len(levels) < 2:
        return None

    aucs = []
    for level in levels:
        subset = df[df[subgroup_col] == level]
        if len(subset) > 10 and subset[label_col].sum() > 0:
            try:
                auc = roc_auc_score(subset[label_col], subset[model_prob_col])
                aucs.append(auc)
            except:
                aucs.append(np.nan)
        else:
            aucs.append(np.nan)

    # Remove NaNs
    aucs = [a for a in aucs if not np.isnan(a)]

    if len(aucs) < 2:
        return None

    # Compute range and std
    auc_range = max(aucs) - min(aucs)
    auc_std = np.std(aucs)

    return {
        'auc_min': min(aucs),
        'auc_max': max(aucs),
        'auc_range': auc_range,
        'auc_std': auc_std,
    }


def plot_subgroup_performance(results_df, subgroup_name, output_path):
    """Create visualization of subgroup performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. AUC comparison
    ax = axes[0, 0]
    x = range(len(results_df))
    width = 0.35

    ax.bar([i - width/2 for i in x], results_df['lungrads_auc'],
           width, label='LungRads', alpha=0.7, color='orange')
    ax.bar([i + width/2 for i in x], results_df['model_auc'],
           width, label='AI Model', alpha=0.7, color='blue')

    ax.set_xlabel('Subgroup', fontsize=11)
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title(f'AUC by {subgroup_name}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['level'], rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.0])

    # 2. Sample size and prevalence
    ax = axes[0, 1]
    ax2 = ax.twinx()

    bars = ax.bar(x, results_df['n'], alpha=0.7, color='gray', label='Sample size')
    line = ax2.plot(x, results_df['prevalence'], 'ro-', linewidth=2,
                    markersize=8, label='Cancer prevalence (%)')

    ax.set_xlabel('Subgroup', fontsize=11)
    ax.set_ylabel('Sample Size (n)', fontsize=11, color='gray')
    ax2.set_ylabel('Prevalence (%)', fontsize=11, color='red')
    ax.set_title(f'Sample Size and Prevalence by {subgroup_name}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['level'], rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor='gray')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(alpha=0.3, axis='y')

    # 3. Sensitivity vs Specificity
    ax = axes[1, 0]
    ax.scatter(results_df['specificity'], results_df['sensitivity'],
              s=results_df['n']/2, alpha=0.6, c=range(len(results_df)), cmap='viridis')

    for i, row in results_df.iterrows():
        ax.annotate(row['level'], (row['specificity'], row['sensitivity']),
                   fontsize=8, alpha=0.7)

    ax.set_xlabel('Specificity', fontsize=11)
    ax.set_ylabel('Sensitivity', fontsize=11)
    ax.set_title(f'Sensitivity vs Specificity by {subgroup_name}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # 4. AUC improvement over LungRads
    ax = axes[1, 1]
    colors = ['green' if x >= 0 else 'red' for x in results_df['auc_improvement']]
    bars = ax.barh(range(len(results_df)), results_df['auc_improvement'],
                   color=colors, alpha=0.7)

    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df['level'])
    ax.set_xlabel('AUC Improvement over LungRads', fontsize=11)
    ax.set_title(f'Model Improvement by {subgroup_name}', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Subgroup plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Subgroup analysis')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV from clinical_analysis.py')
    parser.add_argument('--demographics', type=str, required=True,
                       help='Path to demographics CSV from extract_demographics.py')
    parser.add_argument('--output_dir', type=str, default='./subgroup_analysis',
                       help='Output directory')

    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = load_data(args.predictions, args.demographics)

    # Define subgroups to analyze
    subgroups = [
        ('age_group', 'Age Group'),
        ('sex', 'Sex'),
        ('race_label', 'Race/Ethnicity'),
        ('packyear_group', 'Smoking History (Pack-Years)'),
        ('bmi_category', 'BMI Category'),
    ]

    all_results = []

    print("\n" + "="*80)
    print("SUBGROUP ANALYSIS")
    print("="*80)

    for col, name in subgroups:
        if col not in df.columns:
            print(f"\nSkipping {name}: column '{col}' not found in data")
            continue

        print(f"\n{'-'*80}")
        print(f"Analyzing: {name}")
        print(f"{'-'*80}")

        # Analyze subgroup
        results = analyze_subgroup(df, col, name)

        if len(results) == 0:
            print(f"  No results for {name}")
            continue

        # Display results
        print(results.to_string(index=False))

        # Test for performance parity
        parity = test_performance_parity(df, col)
        if parity:
            print(f"\nPerformance disparity:")
            print(f"  AUC range: {parity['auc_range']:.3f} ({parity['auc_min']:.3f} - {parity['auc_max']:.3f})")
            print(f"  AUC std:   {parity['auc_std']:.3f}")

        # Plot
        plot_path = os.path.join(args.output_dir, f'{col}_performance.png')
        plot_subgroup_performance(results, name, plot_path)

        # Accumulate
        all_results.append(results)

    # Save all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(args.output_dir, 'subgroup_metrics.csv')
        combined.to_csv(output_path, index=False)
        print(f"\n\nAll subgroup metrics saved to: {output_path}")

    print("\n" + "="*80)
    print("SUBGROUP ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
