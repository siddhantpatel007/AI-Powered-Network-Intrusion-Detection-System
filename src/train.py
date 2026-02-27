"""
Phase 5.1 — Model Training & Evaluation
Trains 4 ML models on both binary and multi-class classification tasks.
Generates confusion matrices, classification reports, and comparison metrics.

HOW TO RUN:
    cd "C:\Capstone Project"
    python src/train.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import logging
import time
import gc
import sys, os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# DATA LOADING
# ============================================================
def load_data() -> dict:
    """
    Load all train/val/test splits.

    Returns:
        Dict with X_train, y_train_binary, y_train_multi, etc.
    """
    logger.info("Loading data splits...")

    # Check if SMOTE data exists
    smote_path = DATA_PROCESSED / "X_train_smote_scaled.csv"
    use_smote = smote_path.exists()

    if use_smote:
        logger.info("  Using SMOTE training data")
        X_train = pd.read_csv(smote_path)
        y_train = pd.read_csv(DATA_PROCESSED / "y_train_smote.csv")
        y_train_multi = y_train['Label_Multi']
        # For SMOTE, we need to create binary labels from multi
        y_train_binary = (y_train_multi != 0).astype(int)
        # Assuming class 0 is BENIGN — verify from label encoder
    else:
        logger.info("  Using original training data (no SMOTE)")
        X_train = pd.read_csv(DATA_PROCESSED / "X_train_scaled.csv")
        y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv")
        y_train_multi = y_train['Label_Multi']
        y_train_binary = y_train['Label_Binary']

    X_val = pd.read_csv(DATA_PROCESSED / "X_val_scaled.csv")
    y_val = pd.read_csv(DATA_PROCESSED / "y_val.csv")

    X_test = pd.read_csv(DATA_PROCESSED / "X_test_scaled.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv")

    data = {
        'X_train': X_train,
        'y_train_binary': y_train_binary,
        'y_train_multi': y_train_multi,
        'X_val': X_val,
        'y_val_binary': y_val['Label_Binary'],
        'y_val_multi': y_val['Label_Multi'],
        'X_test': X_test,
        'y_test_binary': y_test['Label_Binary'],
        'y_test_multi': y_test['Label_Multi'],
        'use_smote': use_smote,
    }

    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val:   {X_val.shape}")
    logger.info(f"  Test:  {X_test.shape}")

    return data


# ============================================================
# TRAIN AND EVALUATE ONE MODEL
# ============================================================
def train_and_evaluate(model, model_name: str, X_train, y_train,
                       X_test, y_test, task: str, label_encoder=None) -> dict:
    """
    Train a model and evaluate it.

    Args:
        model: sklearn-compatible classifier.
        model_name: Name for saving files.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        task: 'binary' or 'multi'.
        label_encoder: Optional LabelEncoder for class names.

    Returns:
        Dict of metrics.
    """
    logger.info(f"  Training {model_name} ({task})...")
    t0 = time.time()

    # Train
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # ROC-AUC
    roc_auc = None
    try:
        y_proba = model.predict_proba(X_test)
        if task == 'binary':
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except Exception as e:
        logger.warning(f"    ROC-AUC failed: {e}")

    logger.info(f"    Accuracy: {acc:.4f} | F1(w): {f1_w:.4f} | F1(m): {f1_m:.4f} | Time: {train_time:.1f}s")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8) if task == 'binary' else (14, 12))

    if task == 'multi' and label_encoder is not None:
        # Get actual classes present in test set
        classes_in_test = sorted(y_test.unique())
        class_names = [label_encoder.inverse_transform([c])[0] for c in classes_in_test]
        # Recompute CM with only present classes
        cm = confusion_matrix(y_test, y_pred, labels=classes_in_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
    else:
        labels = ['Benign', 'Attack'] if task == 'binary' else None
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{model_name} — {task.title()} Confusion Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()

    cm_path = REPORTS_DIR / f"{model_name}_{task}_cm.png"
    fig.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Classification Report
    report = classification_report(y_test, y_pred, zero_division=0)
    report_path = REPORTS_DIR / f"{model_name}_{task}_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"{model_name} — {task.title()} Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    # Save model
    model_path = MODELS_DIR / f"{model_name}_{task}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"    Model saved: {model_path}")

    result = {
        'model_name': model_name,
        'task': task,
        'accuracy': round(acc, 4),
        'precision_weighted': round(prec, 4),
        'recall_weighted': round(rec, 4),
        'f1_weighted': round(f1_w, 4),
        'f1_macro': round(f1_m, 4),
        'roc_auc': round(roc_auc, 4) if roc_auc else None,
        'training_time_sec': round(train_time, 1),
    }
    return result


# ============================================================
# RUN ALL MODELS
# ============================================================
def run_all_models(data: dict) -> pd.DataFrame:
    """
    Train and evaluate all 4 models on both tasks.

    Returns:
        DataFrame with all results.
    """
    # Load label encoder for class names in confusion matrix
    le_path = MODELS_DIR / "label_encoder.joblib"
    le = joblib.load(le_path) if le_path.exists() else None

    # Define models
    use_balanced = not data['use_smote']
    cw = 'balanced' if use_balanced else None
    logger.info(f"Using class_weight={'balanced' if use_balanced else 'None (SMOTE applied)'}")

    models = {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=20, random_state=RANDOM_SEED, class_weight=cw
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1, class_weight=cw
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=7,
            random_state=RANDOM_SEED, use_label_encoder=False,
            eval_metric='mlogloss', n_jobs=-1, tree_method='hist'
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1, class_weight=cw
        ),
    }

    results = []

    for name, model in models.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"MODEL: {name}")
        logger.info(f"{'='*40}")

        # Binary task
        import copy
        model_binary = copy.deepcopy(model)
        if name == "XGBoost":
            model_binary.set_params(eval_metric='logloss', objective='binary:logistic')

        res_binary = train_and_evaluate(
            model_binary, name,
            data['X_train'], data['y_train_binary'],
            data['X_test'], data['y_test_binary'],
            task='binary'
        )
        results.append(res_binary)
        del model_binary; gc.collect()

        # Multi-class task
        model_multi = copy.deepcopy(model)
        if name == "XGBoost":
            model_multi.set_params(eval_metric='mlogloss', objective='multi:softprob')
            model_multi.set_params(num_class=data['y_train_multi'].nunique())

        res_multi = train_and_evaluate(
            model_multi, name,
            data['X_train'], data['y_train_multi'],
            data['X_test'], data['y_test_multi'],
            task='multi', label_encoder=le
        )
        results.append(res_multi)
        del model_multi; gc.collect()

    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    results_path = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved: {results_path}")

    return results_df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 5.1: Model Training & Evaluation")
    logger.info("=" * 60)

    # Load data
    data = load_data()

    # Train all models
    results_df = run_all_models(data)

    # Print comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)

    for task in ['binary', 'multi']:
        print(f"\n--- {task.upper()} CLASSIFICATION ---")
        task_df = results_df[results_df['task'] == task].sort_values('f1_weighted', ascending=False)
        print(task_df[['model_name', 'accuracy', 'f1_weighted', 'f1_macro',
                       'roc_auc', 'training_time_sec']].to_string(index=False))

    # Identify best model
    multi_df = results_df[results_df['task'] == 'multi'].sort_values('f1_weighted', ascending=False)
    best = multi_df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best['model_name']} (Multi-class F1_weighted: {best['f1_weighted']})")

    elapsed = time.time() - start
    logger.info(f"\n✅ All training complete in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")