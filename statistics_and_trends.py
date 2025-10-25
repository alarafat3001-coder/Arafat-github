"""
statistics_and_trends.py

Author: Md Al Arafat
Student ID: 23088114

This script:
- Cleans and analyzes a health and lifestyle dataset
- Computes the four main statistical moments
- Produces three plots (relational, categorical, and statistical)

Libraries: pandas, numpy, scipy, seaborn, matplotlib

Usage Example:
    python statistics_and_trends.py --input health_lifestyle_dataset.csv --outdir outputs
"""
# Import Required Libraries

import os, argparse
from pathlib import Path
import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from scipy import stats

# Load Data

def load_data(path: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df

# Clean Data

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset by removing duplicates, handling missing values, and converting categorical columns."""
    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Dropped {before - after} duplicate rows.")

    # Convert gender column to categorical type if it exists
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype('category')

    # Count missing values
    na_count = df.isna().sum().sum()
    print(f"Total missing values in dataset: {na_count}")

    # Drop rows with missing values (if any)
    if na_count > 0:
        df = df.dropna()
        print(f"Dropped rows with missing values; new length = {len(df)}")

    return df

# Compute Statistical Moments

def compute_moments(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Compute mean, variance, skewness, and kurtosis for selected numeric columns."""
    records = []
    for c in cols:
        series = df[c].dropna()
        mean = series.mean()  # First moment (Mean)
        variance = series.var(ddof=0)  # Second moment (Variance)
        skewness = stats.skew(series)  # Third moment (Skewness)
        kurt = stats.kurtosis(series, fisher=False)  # Fourth moment (Kurtosis)
        records.append({
            'feature': c,
            'mean': mean,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurt,
        })
    moments_df = pd.DataFrame.from_records(records)
    return moments_df

# Categorical Plot

def plot_categorical(df: pd.DataFrame, out_path: Path):
    """Create a bar plot showing mean disease_risk by smoker and alcohol consumption."""
    plt.figure(figsize=(8, 6))
    sns.set(style='whitegrid')

    # Ensure smoker and alcohol are treated as categorical variables
    df['smoker'] = df['smoker'].astype('category')
    df['alcohol'] = df['alcohol'].astype('category')

    # Calculate average disease risk for each combination
    agg = df.groupby(['smoker', 'alcohol'], observed=True)['disease_risk'].mean().reset_index()

    # Create labels for readability
    agg['smoker_label'] = agg['smoker'].apply(lambda x: 'Non-smoker' if str(x) in ['0', '0.0'] else 'Smoker')
    agg['alcohol_label'] = agg['alcohol'].apply(lambda x: 'No alcohol' if str(x) in ['0', '0.0'] else 'Alcohol')

    # Create the bar plot
    sns.barplot(data=agg, x='smoker_label', y='disease_risk', hue='alcohol_label')
    plt.ylabel('Mean Disease Risk')
    plt.xlabel('Smoker Status')
    plt.title('Mean Disease Risk by Smoker and Alcohol Consumption')
    plt.legend(title='Alcohol')
    plt.tight_layout()

    # Save the figure
    filepath = out_path / 'categorical_disease_risk_by_smoker_alcohol.png'
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved categorical plot to {filepath}")


# Correlation Heatmap

def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path, cols: list):
    """Create a correlation heatmap for numeric health indicators."""
    plt.figure(figsize=(10, 8))
    sns.set(style='white')

    # Calculate correlation matrix
    corr = df[cols].corr()

    # Create heatmap with annotations
    sns.heatmap(corr, annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={'shrink': .75})
    plt.title('Correlation Heatmap of Health Indicators')
    plt.tight_layout()

    # Save the figure
    filepath = out_path / 'correlation_heatmap.png'
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {filepath}")

# Relational Plot

def plot_relational(df: pd.DataFrame, out_path: Path):
    """Create a scatter plot showing daily steps vs BMI, colored by gender."""
    plt.figure(figsize=(8, 6))
    sns.set(style='whitegrid')  # Use Seaborn styling for better visuals

    # Sample 10% of data for faster plotting
    sns.scatterplot(data=df.sample(frac=0.1, random_state=1), x='bmi', y='daily_steps', hue='gender', alpha=0.6)
    plt.xlabel('BMI')
    plt.ylabel('Daily Steps')
    plt.title('Daily Steps vs BMI (sampled 10%)')
    plt.tight_layout()

    # Save the figure
    filepath = out_path / 'relational_dailysteps_bmi.png'
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved relational plot to {filepath}")


# Main Function

def main(args):
    # Read input path and output directory
    input_path = args.input
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Data
    df = load_data(input_path)
    print(f"Loaded data with shape: {df.shape}")

    # Step 2: Clean Data
    df = clean_data(df)

    # Step 3: Select numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'id']  # Exclude ID column

    # Choose relevant columns for plotting and statistical moments
    chosen_cols = [
        col for col in ['age', 'bmi', 'daily_steps', 'sleep_hours', 'water_intake_l',
                       'calories_consumed', 'resting_hr', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'disease_risk']
        if col in df.columns
    ]

    # Step 4: Compute Statistical Moments
    moments_df = compute_moments(df, chosen_cols)
    print('\nFirst Four Statistical Moments:')
    print(moments_df.to_string(index=False, float_format='{:0.4f}'.format))

    # Save moments as CSV for documentation
    moments_csv = outdir / 'statistical_moments.csv'
    moments_df.to_csv(moments_csv, index=False)
    print(f"Saved statistical moments to {moments_csv}")

    # Step 5: Generate Plots
    plot_relational(df, outdir)  # Relational plot
    plot_categorical(df, outdir)  # Categorical plot
    plot_correlation_heatmap(df, outdir, chosen_cols)  # Statistical heatmap

    print('\nAll done! Outputs saved in:', outdir)

# Run Script

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistics & Trends assignment script')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--outdir', type=str, default='outputs', help='Directory to save plots and CSVs')
    args = parser.parse_args()
    main(args)
    