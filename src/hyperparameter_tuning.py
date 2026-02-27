"""
Phase 5.2 — Hyperparameter Tuning
Tunes the top 2 performing models using RandomizedSearchCV.

HOW TO RUN:
    cd "C:\Capstone Project"
    python src/hyperparameter_tuning.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
import time
import gc
import sys, os
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# PARAM GRIDS
# ============================================================
PARAM_GRIDS = {
    "RandomForest": {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
    },
    "DecisionTree": {
        'max_depth': [10, 15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy'],
    },
    "LogisticRegression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
    },
}


def get_base_model(name: str, n_classes: int):
    """Return a fresh base model instance by name."""
    if name == "RandomForest":
        return RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced')
    elif name == "XGBoost":
        return XGBClassifier(
            random_state=RANDOM_SEED, n_jobs=-1, use_label_encoder=False,
            eval_metric='mlogloss', tree_method='hist', num_class=n_classes
        )
    elif name == "DecisionTree":
        return DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight='balanced')
    elif name == "LogisticRegression":
        return LogisticRegression(random_state=RANDOM_SEED, n_jobs=-1,
                                  max_iter=1000, class_weight='balanced')
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 5.2: Hyperparameter Tuning")
    logger.info("=" * 60)

    # Load results to find top 2 models
    results = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    multi_results = results[results['task'] == 'multi'].sort_values('f1_weighted', ascending=False)
    top2_names = multi_results['model_name'].head(2).tolist()
    logger.info(f"Top 2 models to tune: {top2_names}")

    # Load training and test data
    smote_path = DATA_PROCESSED / "X_train_smote_scaled.csv"
    if smote_path.exists():
        X_train = pd.read_csv(smote_path)
        y_train = pd.read_csv(DATA_PROCESSED / "y_train_smote.csv")['Label_Multi']
    else:
        X_train = pd.read_csv(DATA_PROCESSED / "X_train_scaled.csv")
        y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv")['Label_Multi']

    X_test = pd.read_csv(DATA_PROCESSED / "X_test_scaled.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv")['Label_Multi']

    n_classes = y_train.nunique()

    # Sample for tuning if too large
    if len(X_train) > 500000:
        logger.info(f"Sampling 500k rows for tuning (original: {len(X_train):,})")
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train, train_size=500000,
            stratify=y_train, random_state=RANDOM_SEED
        )
    else:
        X_tune, y_tune = X_train, y_train

    # Tune each model
    tuning_results = []

    for name in top2_names:
        logger.info(f"\n{'='*50}")
        logger.info(f"Tuning: {name}")
        logger.info(f"{'='*50}")

        if name not in PARAM_GRIDS:
            logger.warning(f"No param grid for {name}, skipping")
            continue

        base_model = get_base_model(name, n_classes)
        param_grid = PARAM_GRIDS[name]

        # Original F1
        orig_model = get_base_model(name, n_classes)
        orig_model.fit(X_tune, y_tune)
        orig_pred = orig_model.predict(X_test)
        orig_f1 = f1_score(y_test, orig_pred, average='weighted', zero_division=0)
        logger.info(f"  Original F1 (weighted): {orig_f1:.4f}")
        del orig_model; gc.collect()

        # Randomized search
        search = RandomizedSearchCV(
            base_model, param_grid,
            n_iter=15, cv=3, scoring='f1_weighted',
            random_state=RANDOM_SEED, n_jobs=-1, verbose=1
        )

        t0 = time.time()
        search.fit(X_tune, y_tune)
        search_time = time.time() - t0

        logger.info(f"  Best CV score: {search.best_score_:.4f}")
        logger.info(f"  Best params: {search.best_params_}")
        logger.info(f"  Search time: {search_time:.1f}s")

        # Re-train on full training data with best params
        logger.info(f"  Re-training best model on full training data...")
        best_model = search.best_estimator_
        best_model.fit(X_train, y_train)

        # Evaluate on test
        y_pred = best_model.predict(X_test)
        tuned_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        improvement = ((tuned_f1 - orig_f1) / orig_f1) * 100
        logger.info(f"  Original F1: {orig_f1:.4f} → Tuned F1: {tuned_f1:.4f} ({improvement:+.2f}%)")

        # Save
        model_path = MODELS_DIR / f"{name}_tuned_multi.joblib"
        joblib.dump(best_model, model_path)
        logger.info(f"  Saved: {model_path}")

        params_path = MODELS_DIR / f"{name}_best_params.json"
        # Convert numpy types to Python types for JSON
        clean_params = {k: int(v) if isinstance(v, (np.integer,)) else
                        float(v) if isinstance(v, (np.floating,)) else
                        v for k, v in search.best_params_.items()}
        with open(params_path, 'w') as f:
            json.dump(clean_params, f, indent=2)

        tuning_results.append({
            'model_name': name,
            'original_f1': orig_f1,
            'tuned_f1': tuned_f1,
            'improvement_pct': improvement,
            'best_params': clean_params,
            'search_time_sec': search_time,
        })

        del best_model, search; gc.collect()

    # Pick overall best
    if tuning_results:
        best_result = max(tuning_results, key=lambda x: x['tuned_f1'])
        best_name = best_result['model_name']

        # Load and save as "best_model"
        best_model_path = MODELS_DIR / f"{best_name}_tuned_multi.joblib"
        best_model = joblib.load(best_model_path)
        joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

        with open(MODELS_DIR / "best_model_name.txt", 'w') as f:
            f.write(best_name)

        with open(MODELS_DIR / "best_params.json", 'w') as f:
            json.dump(best_result['best_params'], f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("=" * 60)
        for r in tuning_results:
            print(f"\n{r['model_name']}:")
            print(f"  Original F1: {r['original_f1']:.4f}")
            print(f"  Tuned F1:    {r['tuned_f1']:.4f}")
            print(f"  Improvement: {r['improvement_pct']:+.2f}%")
            print(f"  Best params: {r['best_params']}")

        print(f"\n🏆 BEST OVERALL: {best_name} (F1: {best_result['tuned_f1']:.4f})")
        print(f"   Saved as: models/best_model.joblib")

    elapsed = time.time() - start
    logger.info(f"\n✅ Tuning complete in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")