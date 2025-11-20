#!/usr/bin/env python3
"""
Extract demographic information from NLST metadata for subgroup analysis.

This script loads the NLST metadata and extracts:
- Age
- Sex
- Race/ethnicity
- Smoking history (pack-years)
- BMI
- Other clinical covariates

Usage:
    python extract_demographics.py --metadata_path PATH --output demographics.csv
"""

import argparse
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_demographics_from_metadata(metadata_path):
    """
    Extract demographic information from NLST metadata JSON.

    The metadata structure includes patient-level info in pt_metadata field.
    """
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    demographics = []

    for mrn_row in tqdm(metadata, desc="Extracting demographics"):
        pid = mrn_row["pid"]
        pt_metadata = mrn_row["pt_metadata"]

        # Extract demographic fields
        # Note: Field names are based on NLST data dictionary
        # Adjust based on actual metadata structure
        demo = {
            'pid': pid,
            'age': pt_metadata.get('age', [np.nan])[0],
            'gender': pt_metadata.get('gender', [np.nan])[0],  # 1=male, 2=female
            'race': pt_metadata.get('race', [np.nan])[0],  # 1=White, 2=Black, etc.
            'ethnic': pt_metadata.get('ethnic', [np.nan])[0],  # Ethnicity
            'bmi': pt_metadata.get('bmi', [np.nan])[0],
            'pkyr': pt_metadata.get('pkyr', [np.nan])[0],  # Pack-years smoking
            'smokeage': pt_metadata.get('smokeage', [np.nan])[0],  # Age started smoking
            'smokeday': pt_metadata.get('smokeday', [np.nan])[0],  # Cigarettes per day
            'diagyr': pt_metadata.get('diagyr', [np.nan])[0],  # Year of cancer diagnosis
            'diagage': pt_metadata.get('diagage', [np.nan])[0],  # Age at cancer diagnosis
            'candx_days': pt_metadata.get('candx_days', [np.nan])[0],  # Days to cancer diagnosis
            'fup_days': pt_metadata.get('fup_days', [np.nan])[0],  # Days of follow-up
            'de_death': pt_metadata.get('de_death', [np.nan])[0],  # Death indicator
        }

        demographics.append(demo)

    df = pd.DataFrame(demographics)

    # Create categorical labels
    df['sex'] = df['gender'].map({1: 'Male', 2: 'Female'})
    df['race_label'] = df['race'].map({
        1: 'White',
        2: 'Black',
        3: 'Asian',
        4: 'Native American',
        5: 'Pacific Islander',
        6: 'Multiple',
        7: 'Unknown'
    })

    # Create age groups
    df['age_group'] = pd.cut(df['age'],
                             bins=[0, 60, 65, 70, 75, 100],
                             labels=['<60', '60-65', '65-70', '70-75', '75+'])

    # Create pack-year groups
    df['packyear_group'] = pd.cut(df['pkyr'],
                                   bins=[0, 30, 50, 100, 200],
                                   labels=['<30', '30-50', '50-100', '100+'])

    # Create BMI categories
    df['bmi_category'] = pd.cut(df['bmi'],
                                bins=[0, 18.5, 25, 30, 100],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    return df


def main():
    parser = argparse.ArgumentParser(description='Extract NLST demographics')
    parser.add_argument('--metadata_path', type=str,
                       default='../../../data/project2/nlst-metadata/full_nlst_google.json',
                       help='Path to NLST metadata JSON')
    parser.add_argument('--output', type=str,
                       default='./demographics.csv',
                       help='Output CSV file')

    args = parser.parse_args()

    # Extract demographics
    df = extract_demographics_from_metadata(args.metadata_path)

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nDemographics saved to: {args.output}")

    # Print summary statistics
    print("\n" + "="*80)
    print("DEMOGRAPHIC SUMMARY")
    print("="*80)
    print(f"\nTotal patients: {len(df)}")

    print("\nSex distribution:")
    print(df['sex'].value_counts())

    print("\nRace distribution:")
    print(df['race_label'].value_counts())

    print("\nAge distribution:")
    print(df['age_group'].value_counts().sort_index())

    print("\nPack-year distribution:")
    print(df['packyear_group'].value_counts().sort_index())

    print("\nBMI distribution:")
    print(df['bmi_category'].value_counts().sort_index())


if __name__ == '__main__':
    main()
