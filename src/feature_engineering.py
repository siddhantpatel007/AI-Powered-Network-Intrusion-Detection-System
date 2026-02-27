"""
Phase 3 — Feature Engineering & Selection
Removes constant/correlated features, selects top 30 via Random Forest importance.

HOW TO RUN:
    cd "C:\Capstone Project"
    python src/feature_engineering.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import logging
import time
import sys, os
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return only numeric feature columns (exclude labels)."""
    exclude = ['Label', 'Label_Binary', 'Label_Multi']
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]


def remove_constant_features(df: pd.DataFrame, threshold: float = 0.01) -> tuple:
    """
    Remove features with near-zero variance.

    Args:
        df: DataFrame.
        threshold: Variance threshold below which features are removed.

    Returns:
        Tuple of (filtered DataFrame, list of removed column names).
    """
    feature_cols = get_feature_columns(df)
    logger.info(f"Starting with {len(feature_cols)} numeric features")

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[feature_cols])

    mask = selector.get_support()
    removed = [feature_cols[i] for i in range(len(feature_cols)) if not mask[i]]
    kept = [feature_cols[i] for i in range(len(feature_cols)) if mask[i]]

    logger.info(f"Removed {len(removed)} constant/near-zero variance features: {removed}")
    logger.info(f"Remaining: {len(kept)} features")

    # Drop removed columns from df
    df = df.drop(columns=removed)
    return df, removed


def remove_highly_correlated(df: pd.DataFrame, threshold: float = 0.95) -> tuple:
    """
    Remove one of each pair of highly correlated features.

    Args:
        df: DataFrame.
        threshold: Correlation threshold above which one feature is dropped.

    Returns:
        Tuple of (filtered DataFrame, list of removed column names).
    """
    feature_cols = get_feature_columns(df)
    logger.info(f"Checking correlations among {len(feature_cols)} features (threshold={threshold})")

    corr_matrix = df[feature_cols].corr().abs()

    # Get upper triangle to avoid duplicates
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find pairs above threshold
    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        if correlated:
            for corr_col in correlated:
                # Drop the one with lower correlation to target
                if 'Label_Binary' in df.columns:
                    corr_to_target_col = abs(df[col].corr(df['Label_Binary']))
                    corr_to_target_corr = abs(df[corr_col].corr(df['Label_Binary']))
                    drop = corr_col if corr_to_target_col >= corr_to_target_corr else col
                else:
                    drop = corr_col
                to_drop.add(drop)

    to_drop = list(to_drop)
    logger.info(f"Removed {len(to_drop)} highly correlated features")
    if to_drop:
        logger.info(f"  Dropped: {to_drop[:10]}{'...' if len(to_drop) > 10 else ''}")

    df = df.drop(columns=to_drop)
    return df, to_drop


def select_top_features_rf(X: pd.DataFrame, y: pd.Series,
                           n_features: int = 30) -> tuple:
    """
    Select top N features using Random Forest importance.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        n_features: Number of features to select.

    Returns:
        Tuple of (list of top feature names, Series of importances).
    """
    logger.info(f"Selecting top {n_features} features using Random Forest...")

    # Sample if dataset is too large
    if len(X) > 300000:
        logger.info(f"  Sampling 300,000 rows for RF (original: {len(X):,})")
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=300000, stratify=y, random_state=RANDOM_SEED
        )
    else:
        X_sample, y_sample = X, y

    # Train quick Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_SEED,
        n_jobs=-1, max_depth=20
    )
    rf.fit(X_sample, y_sample)

    # Get importances
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    top_features = importances.head(n_features).index.tolist()
    logger.info(f"  Top 5 features: {top_features[:5]}")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(12, 10))
    top_imp = importances.head(n_features).sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_imp)))
    ax.barh(range(len(top_imp)), top_imp.values, color=colors)
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp.index, fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {n_features} Features by Random Forest Importance",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = REPORTS_DIR / "feature_importance_rf.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Chart saved: {path}")

    return top_features, importances


def create_final_dataset(df: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    """
    Create final dataset with only selected features + labels.

    Args:
        df: Full DataFrame.
        selected_features: List of feature column names to keep.

    Returns:
        pd.DataFrame: Final dataset.
    """
    label_cols = ['Label', 'Label_Binary', 'Label_Multi']
    keep_cols = selected_features + [c for c in label_cols if c in df.columns]
    df_final = df[keep_cols].copy()

    # Save dataset
    output_path = DATA_PROCESSED / "cicids2017_final.csv"
    df_final.to_csv(output_path, index=False)
    logger.info(f"Final dataset saved: {output_path}")
    logger.info(f"  Shape: {df_final.shape}")

    # Save feature list
    features_path = MODELS_DIR / "selected_features.json"
    with open(features_path, 'w') as f:
        json.dump(selected_features, f, indent=2)
    logger.info(f"Feature list saved: {features_path}")

    return df_final


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 3: Feature Engineering & Selection")
    logger.info("=" * 60)

    # Load cleaned data
    df = pd.read_csv(DATA_PROCESSED / "cicids2017_cleaned.csv", low_memory=False)
    initial_features = len(get_feature_columns(df))
    logger.info(f"Loaded data. Shape: {df.shape}, Features: {initial_features}")

    # Step 1: Remove constant features
    df, removed_constant = remove_constant_features(df, threshold=0.01)
    after_constant = len(get_feature_columns(df))

    # Step 2: Remove highly correlated features
    df, removed_correlated = remove_highly_correlated(df, threshold=0.95)
    after_correlated = len(get_feature_columns(df))

    # Step 3: Select top 30 features with RF
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df['Label_Multi']
    top_features, importances = select_top_features_rf(X, y, n_features=30)

    # Step 4: Create final dataset
    df_final = create_final_dataset(df, top_features)

    # Summary
    print("\n" + "=" * 60)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 60)
    print(f"Initial features:              {initial_features}")
    print(f"After constant removal:        {after_constant} (removed {initial_features - after_constant})")
    print(f"After correlation removal:     {after_correlated} (removed {after_constant - after_correlated})")
    print(f"After RF selection:            {len(top_features)}")
    print(f"\nFinal dataset shape: {df_final.shape}")
    print(f"\nSelected features:")
    for i, feat in enumerate(top_features, 1):
        imp = importances.get(feat, 0)
        print(f"  {i:2d}. {feat:40s} (importance: {imp:.4f})")

    elapsed = time.time() - start
    logger.info(f"\n✅ Feature engineering complete in {elapsed:.1f} seconds")