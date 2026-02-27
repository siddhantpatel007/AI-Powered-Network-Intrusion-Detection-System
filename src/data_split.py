"""
Phase 4 — Data Splitting, SMOTE, and Scaling
Splits data into train/val/test, applies SMOTE to training set, scales all sets.

HOW TO RUN:
    cd "C:\Capstone Project"
    python src/data_split.py
"""

import pandas as pd
import numpy as np
import joblib
import logging
import time
import gc
import sys, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED, MODELS_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_final_data() -> pd.DataFrame:
    """Load the feature-selected dataset."""
    path = DATA_PROCESSED / "cicids2017_final.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded. Shape: {df.shape}")
    return df


def split_features_labels(df: pd.DataFrame) -> tuple:
    """
    Separate features from labels.

    Returns:
        Tuple of (X, y_binary, y_multi, label_original)
    """
    label_cols = ['Label', 'Label_Binary', 'Label_Multi']
    feature_cols = [c for c in df.columns if c not in label_cols]

    X = df[feature_cols]
    y_binary = df['Label_Binary']
    y_multi = df['Label_Multi']
    label_original = df['Label']

    logger.info(f"Features: {X.shape[1]} columns")
    logger.info(f"Samples: {len(X):,}")
    return X, y_binary, y_multi, label_original


def create_splits(X, y_binary, y_multi):
    """
    Create 70/15/15 stratified train/val/test splits.

    Returns:
        Dict with all split arrays.
    """
    logger.info("Creating 70/15/15 stratified splits...")

    # First split: 70% train, 30% temp
    X_train, X_temp, yb_train, yb_temp, ym_train, ym_temp = train_test_split(
        X, y_binary, y_multi,
        test_size=(TEST_SIZE + VAL_SIZE),
        stratify=y_multi,
        random_state=RANDOM_SEED
    )

    # Second split: temp into 50/50 → val and test
    X_val, X_test, yb_val, yb_test, ym_val, ym_test = train_test_split(
        X_temp, yb_temp, ym_temp,
        test_size=0.5,
        stratify=ym_temp,
        random_state=RANDOM_SEED
    )

    logger.info(f"  Train: {X_train.shape} | Benign: {(yb_train==0).sum():,} | Attack: {(yb_train==1).sum():,}")
    logger.info(f"  Val:   {X_val.shape}  | Benign: {(yb_val==0).sum():,} | Attack: {(yb_val==1).sum():,}")
    logger.info(f"  Test:  {X_test.shape}  | Benign: {(yb_test==0).sum():,} | Attack: {(yb_test==1).sum():,}")

    return {
        'X_train': X_train, 'yb_train': yb_train, 'ym_train': ym_train,
        'X_val': X_val, 'yb_val': yb_val, 'ym_val': ym_val,
        'X_test': X_test, 'yb_test': yb_test, 'ym_test': ym_test,
    }


def apply_smote(X_train, ym_train):
    """
    Apply SMOTE to the training set. Falls back to class weights if SMOTE fails.

    Returns:
        Tuple of (X_train_smote, ym_train_smote) or (None, None) if failed.
    """
    logger.info("Applying SMOTE to training data...")
    logger.info(f"  Before SMOTE: {X_train.shape}")

    try:
        from imblearn.over_sampling import SMOTE

        # If training set is very large, sample before SMOTE
        if len(X_train) > 500000:
            logger.info(f"  Training set too large for SMOTE ({len(X_train):,}). Sampling 500k rows.")
            idx = pd.Series(range(len(X_train)))
            sample_idx, _ = train_test_split(
                idx, train_size=500000, stratify=ym_train, random_state=RANDOM_SEED
            )
            X_for_smote = X_train.iloc[sample_idx.values]
            y_for_smote = ym_train.iloc[sample_idx.values]
        else:
            X_for_smote = X_train
            y_for_smote = ym_train

        smote = SMOTE(random_state=RANDOM_SEED, n_jobs=-1)
        X_smote, y_smote = smote.fit_resample(X_for_smote, y_for_smote)

        logger.info(f"  After SMOTE: {X_smote.shape}")
        logger.info(f"  Class distribution after SMOTE:")
        for cls in sorted(y_smote.unique()):
            logger.info(f"    Class {cls}: {(y_smote == cls).sum():,}")

        return pd.DataFrame(X_smote, columns=X_train.columns), pd.Series(y_smote, name='Label_Multi')

    except MemoryError:
        logger.warning("  ⚠️ SMOTE failed due to MemoryError. Will use class_weight='balanced' instead.")
        return None, None
    except Exception as e:
        logger.warning(f"  ⚠️ SMOTE failed: {e}. Will use class_weight='balanced' instead.")
        return None, None


def scale_data(splits: dict, X_smote=None):
    """
    Fit StandardScaler on training data, transform all splits.

    Returns:
        Dict with scaled arrays + scaler saved to disk.
    """
    logger.info("Applying StandardScaler...")

    scaler = StandardScaler()
    feature_cols = splits['X_train'].columns.tolist()

    # Fit on original training data
    scaler.fit(splits['X_train'])

    # Save scaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Scaler saved: {scaler_path}")

    # Transform all splits
    scaled = {}
    for key in ['X_train', 'X_val', 'X_test']:
        arr = scaler.transform(splits[key])
        scaled[key] = pd.DataFrame(arr, columns=feature_cols)
        logger.info(f"  Scaled {key}: {scaled[key].shape}")

    # Scale SMOTE data if available
    if X_smote is not None:
        arr = scaler.transform(X_smote)
        scaled['X_train_smote'] = pd.DataFrame(arr, columns=feature_cols)
        logger.info(f"  Scaled X_train_smote: {scaled['X_train_smote'].shape}")

    return scaled


def save_all(splits: dict, scaled: dict, y_smote=None):
    """Save all split files to disk."""
    logger.info("Saving all split files...")

    # Labels
    pd.DataFrame({
        'Label_Binary': splits['yb_train'],
        'Label_Multi': splits['ym_train']
    }).to_csv(DATA_PROCESSED / "y_train.csv", index=False)

    pd.DataFrame({
        'Label_Binary': splits['yb_val'],
        'Label_Multi': splits['ym_val']
    }).to_csv(DATA_PROCESSED / "y_val.csv", index=False)

    pd.DataFrame({
        'Label_Binary': splits['yb_test'],
        'Label_Multi': splits['ym_test']
    }).to_csv(DATA_PROCESSED / "y_test.csv", index=False)

    # Scaled features
    scaled['X_train'].to_csv(DATA_PROCESSED / "X_train_scaled.csv", index=False)
    scaled['X_val'].to_csv(DATA_PROCESSED / "X_val_scaled.csv", index=False)
    scaled['X_test'].to_csv(DATA_PROCESSED / "X_test_scaled.csv", index=False)

    # SMOTE data if available
    if 'X_train_smote' in scaled and y_smote is not None:
        scaled['X_train_smote'].to_csv(DATA_PROCESSED / "X_train_smote_scaled.csv", index=False)
        pd.DataFrame({'Label_Multi': y_smote}).to_csv(
            DATA_PROCESSED / "y_train_smote.csv", index=False
        )
        logger.info("  SMOTE files saved")

    # List all saved files
    logger.info("\n  Saved files:")
    for f in sorted(DATA_PROCESSED.glob("*.csv")):
        size = f.stat().st_size / (1024 * 1024)
        logger.info(f"    {f.name:40s} ({size:.1f} MB)")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 4: Data Splitting, SMOTE, and Scaling")
    logger.info("=" * 60)

    # Load data
    df = load_final_data()
    X, y_binary, y_multi, _ = split_features_labels(df)
    del df; gc.collect()

    # Create splits
    splits = create_splits(X, y_binary, y_multi)
    del X, y_binary, y_multi; gc.collect()

    # Apply SMOTE
    X_smote, y_smote = apply_smote(splits['X_train'], splits['ym_train'])

    # Scale
    scaled = scale_data(splits, X_smote)

    # Save
    save_all(splits, scaled, y_smote)

    # Summary
    print("\n" + "=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    print(f"Train:       {scaled['X_train'].shape}")
    print(f"Validation:  {scaled['X_val'].shape}")
    print(f"Test:        {scaled['X_test'].shape}")
    if 'X_train_smote' in scaled:
        print(f"Train SMOTE: {scaled['X_train_smote'].shape}")
    else:
        print(f"Train SMOTE: Skipped (will use class_weight='balanced')")
    print(f"\nScaler saved: {MODELS_DIR / 'scaler.joblib'}")

    elapsed = time.time() - start
    logger.info(f"\n✅ Data splitting complete in {elapsed:.1f} seconds")