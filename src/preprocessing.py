"""
Phase 2.1 — Data Loading, Cleaning, and Label Encoding
Reads all 8 CICIDS2017 CSV files, cleans them, encodes labels, saves processed data.

HOW TO RUN:
    cd "C:\Capstone Project"
    python src/preprocessing.py
"""

import pandas as pd
import numpy as np
import logging
import time
import joblib
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add parent directory to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_RAW, DATA_PROCESSED, MODELS_DIR, CSV_FILES

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================
# FUNCTION 1: Load Raw Data
# ============================================================
def load_raw_data() -> pd.DataFrame:
    """
    Load all 8 CICIDS2017 CSV files and concatenate into one DataFrame.

    Returns:
        pd.DataFrame: Combined raw dataset.
    """
    dfs = []
    for i, filename in enumerate(CSV_FILES, 1):
        filepath = DATA_RAW / filename
        logger.info(f"Loading file {i}/{len(CSV_FILES)}: {filename}")
        try:
            df = pd.read_csv(filepath, low_memory=False, encoding='utf-8')
            logger.info(f"  → Shape: {df.shape}")
            dfs.append(df)
        except Exception as e:
            logger.error(f"  ❌ Failed to load {filename}: {e}")
            # Try with latin-1 encoding as fallback
            try:
                df = pd.read_csv(filepath, low_memory=False, encoding='latin-1')
                logger.info(f"  → Loaded with latin-1 encoding. Shape: {df.shape}")
                dfs.append(df)
            except Exception as e2:
                logger.error(f"  ❌ Also failed with latin-1: {e2}")
                continue

    if not dfs:
        raise ValueError("No CSV files were loaded! Check your Dataset folder.")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"✅ Loaded {len(dfs)} files. Combined shape: {combined.shape}")
    return combined


# ============================================================
# FUNCTION 2: Clean Data
# ============================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset: strip column names, drop unnecessary columns,
    handle inf/nan values, remove duplicates.

    Args:
        df: Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info(f"Starting cleaning. Original shape: {df.shape}")

    # Step a: Strip whitespace from column names
    df.columns = df.columns.str.strip()
    logger.info(f"Column names stripped. Columns: {list(df.columns[:5])}...")

    # Step b: Drop non-generalizable columns
    cols_to_drop = ['Flow ID', 'Source IP', 'Source Port',
                    'Destination IP', 'Destination Port', 'Timestamp']
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)
        logger.info(f"Dropped columns: {existing_drops}")

    # Step c: Replace infinity values with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    logger.info("Replaced infinity values with NaN")

    # Step d: Drop rows with NaN
    rows_before = len(df)
    df = df.dropna()
    rows_dropped_nan = rows_before - len(df)
    logger.info(f"Dropped {rows_dropped_nan} rows with NaN values")

    # Step e: Drop duplicates
    rows_before = len(df)
    df = df.drop_duplicates()
    rows_dropped_dup = rows_before - len(df)
    logger.info(f"Dropped {rows_dropped_dup} duplicate rows")

    # Step f: Reset index
    df = df.reset_index(drop=True)
    logger.info(f"✅ Cleaning complete. Final shape: {df.shape}")

    return df


# ============================================================
# FUNCTION 3: Encode Labels
# ============================================================
def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary and multi-class encoded label columns.

    Args:
        df: Cleaned DataFrame with 'Label' column.

    Returns:
        pd.DataFrame: DataFrame with Label_Binary and Label_Multi columns added.
    """
    # Check label column exists
    if 'Label' not in df.columns:
        # Try to find it (might have different casing)
        label_cols = [c for c in df.columns if 'label' in c.lower()]
        if label_cols:
            df = df.rename(columns={label_cols[0]: 'Label'})
            logger.info(f"Renamed '{label_cols[0]}' to 'Label'")
        else:
            raise ValueError("No 'Label' column found in dataset!")

    # Strip whitespace from label values
    df['Label'] = df['Label'].astype(str).str.strip()

    # Binary label: 0 = BENIGN, 1 = Attack
    df['Label_Binary'] = (df['Label'] != 'BENIGN').astype(int)
    benign_count = (df['Label_Binary'] == 0).sum()
    attack_count = (df['Label_Binary'] == 1).sum()
    logger.info(f"Binary labels: BENIGN={benign_count}, Attack={attack_count}")

    # Multi-class label: LabelEncoder
    le = LabelEncoder()
    df['Label_Multi'] = le.fit_transform(df['Label'])

    # Save label encoder
    le_path = MODELS_DIR / "label_encoder.joblib"
    joblib.dump(le, le_path)
    logger.info(f"Label encoder saved to {le_path}")

    # Print mapping
    logger.info("Label encoding mapping:")
    for cls, code in zip(le.classes_, le.transform(le.classes_)):
        count = (df['Label'] == cls).sum()
        logger.info(f"  {code:2d} → {cls} (n={count:,})")

    return df


# ============================================================
# FUNCTION 4: Save Processed Data
# ============================================================
def save_processed_data(df: pd.DataFrame) -> None:
    """
    Save the cleaned and encoded dataset.

    Args:
        df: Processed DataFrame.
    """
    output_path = DATA_PROCESSED / "cicids2017_cleaned.csv"
    df.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ Saved cleaned data to {output_path} ({size_mb:.1f} MB)")
    logger.info(f"   Shape: {df.shape}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("PHASE 2.1: Data Preprocessing Pipeline")
    logger.info("=" * 60)

    # Step 1: Load
    df = load_raw_data()

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Encode labels
    df = encode_labels(df)

    # Step 4: Save
    save_processed_data(df)

    # Print class distribution
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    dist = df['Label'].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {label:40s} {count:>10,} ({pct:5.2f}%)")
    print(f"  {'TOTAL':40s} {len(df):>10,}")

    elapsed = time.time() - start_time
    logger.info(f"\n✅ Preprocessing complete in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")