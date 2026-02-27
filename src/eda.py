"""
Phase 2.2 — Exploratory Data Analysis
Generates visualizations and statistics for the cleaned CICIDS2017 dataset.

HOW TO RUN:
    cd "C:\Capstone Project"
    python src/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data() -> pd.DataFrame:
    """Load the cleaned dataset."""
    path = DATA_PROCESSED / "cicids2017_cleaned.csv"
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded. Shape: {df.shape}")
    return df


def dataset_overview(df: pd.DataFrame) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"\nData types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count}")
    print(f"\nNumeric columns summary (first 10):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
    print(df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']].to_string())


def plot_class_distribution_multi(df: pd.DataFrame) -> None:
    """Create horizontal bar chart of attack type counts."""
    logger.info("Creating multi-class distribution chart...")
    dist = df['Label'].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(dist)), dist.values, color=sns.color_palette("husl", len(dist)))
    ax.set_yticks(range(len(dist)))
    ax.set_yticklabels(dist.index, fontsize=10)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("Distribution of Attack Types in CICIDS2017", fontsize=14, fontweight='bold')

    # Add count labels
    for i, (v, label) in enumerate(zip(dist.values, dist.index)):
        ax.text(v + max(dist.values) * 0.01, i, f"{v:,}", va='center', fontsize=9)

    plt.tight_layout()
    path = REPORTS_DIR / "class_distribution_multi.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_class_distribution_binary(df: pd.DataFrame) -> None:
    """Create pie chart of benign vs attack ratio."""
    logger.info("Creating binary distribution pie chart...")
    counts = df['Label_Binary'].value_counts()
    labels = ['Benign', 'Attack']
    colors = ['#2ecc71', '#e74c3c']

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 14}, pctdistance=0.85
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax.set_title("Benign vs Attack Traffic Ratio", fontsize=14, fontweight='bold')

    # Add counts in legend
    legend_labels = [f"{l}: {c:,}" for l, c in zip(labels, counts.values)]
    ax.legend(legend_labels, loc="lower right", fontsize=12)

    plt.tight_layout()
    path = REPORTS_DIR / "class_distribution_binary.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Create correlation heatmap of top 30 features."""
    logger.info("Creating correlation heatmap...")
    numeric_df = df.select_dtypes(include=[np.number])
    # Exclude label columns
    exclude = ['Label_Binary', 'Label_Multi']
    feature_cols = [c for c in numeric_df.columns if c not in exclude]
    corr = numeric_df[feature_cols].corr()

    # Select top 30 by mean absolute correlation
    mean_abs_corr = corr.abs().mean().sort_values(ascending=False)
    top30 = mean_abs_corr.head(30).index.tolist()

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr.loc[top30, top30], cmap='coolwarm', center=0,
        annot=False, linewidths=0.5, ax=ax, vmin=-1, vmax=1
    )
    ax.set_title("Feature Correlation Heatmap (Top 30)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    path = REPORTS_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Plot histograms of top 8 features by variance, colored by class."""
    logger.info("Creating feature distribution plots...")
    numeric_df = df.select_dtypes(include=[np.number])
    exclude = ['Label_Binary', 'Label_Multi']
    feature_cols = [c for c in numeric_df.columns if c not in exclude]

    variances = numeric_df[feature_cols].var().sort_values(ascending=False)
    top8 = variances.head(8).index.tolist()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    benign = df[df['Label_Binary'] == 0]
    attack = df[df['Label_Binary'] == 1]

    for i, col in enumerate(top8):
        ax = axes[i]
        # Use percentile clipping to avoid extreme outliers
        upper = np.percentile(df[col].dropna(), 99)
        lower = np.percentile(df[col].dropna(), 1)

        ax.hist(benign[col].clip(lower, upper), bins=50, alpha=0.5,
                label='Benign', color='#2ecc71', density=True)
        ax.hist(attack[col].clip(lower, upper), bins=50, alpha=0.5,
                label='Attack', color='#e74c3c', density=True)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle("Top 8 Features by Variance — Distribution by Class",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = REPORTS_DIR / "feature_distributions.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_boxplots(df: pd.DataFrame) -> None:
    """Box plots of key features split by benign vs attack."""
    logger.info("Creating box plots...")
    target_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Packet Length Mean'
    ]
    # Find matching columns (handle slight name differences)
    available = []
    for tf in target_features:
        matches = [c for c in df.columns if tf.lower() in c.lower()]
        if matches:
            available.append(matches[0])
        else:
            logger.warning(f"  Feature '{tf}' not found, skipping")

    if len(available) < 2:
        # Fallback: use first 6 numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ['Label_Binary', 'Label_Multi']
        available = [c for c in numeric_cols if c not in exclude][:6]
        logger.info(f"  Using fallback features: {available}")

    n_plots = min(len(available), 6)
    rows = 2
    cols = 3

    # Sample data for faster plotting
    sample = df.sample(min(50000, len(df)), random_state=42)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(n_plots):
        col = available[i]
        ax = axes[i]
        # Clip outliers for readability
        data = sample[[col, 'Label_Binary']].copy()
        upper = data[col].quantile(0.99)
        data[col] = data[col].clip(upper=upper)

        sns.boxplot(x='Label_Binary', y=col, data=data, ax=ax,
                    palette=['#2ecc71', '#e74c3c'])
        ax.set_xticklabels(['Benign', 'Attack'])
        ax.set_title(col, fontsize=10)

    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Comparison: Benign vs Attack",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = REPORTS_DIR / "benign_vs_attack_boxplots.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print dataset summary statistics."""
    dist = df['Label'].value_counts()
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples:          {len(df):,}")
    print(f"Number of features:     {len(df.select_dtypes(include=[np.number]).columns) - 2}")
    print(f"Number of attack types: {df['Label'].nunique()}")
    print(f"Most common attack:     {dist.index[1]} (n={dist.iloc[1]:,})")
    print(f"Rarest attack:          {dist.index[-1]} (n={dist.iloc[-1]:,})")
    benign_count = (df['Label'] == 'BENIGN').sum()
    rarest_count = dist.iloc[-1]
    print(f"Imbalance ratio:        {benign_count / rarest_count:.0f}:1 (Benign vs rarest)")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 2.2: Exploratory Data Analysis")
    logger.info("=" * 60)

    df = load_data()
    dataset_overview(df)
    plot_class_distribution_multi(df)
    plot_class_distribution_binary(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_boxplots(df)
    print_summary(df)

    elapsed = time.time() - start
    logger.info(f"\n✅ EDA complete in {elapsed:.1f} seconds. Charts saved to {REPORTS_DIR}")