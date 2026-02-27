"""
Phase 6 — Model Explainability with SHAP
Generates SHAP summary, importance, dependence, and individual explanation plots.

HOW TO RUN:
    cd "C:\Capstone Project"
    python src/explainability.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib
import json
import logging
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PHASE 6: SHAP Model Explainability")
    logger.info("=" * 60)

    # Load best model
    model_name_path = MODELS_DIR / "best_model_name.txt"
    if model_name_path.exists():
        with open(model_name_path, 'r') as f:
            model_name = f.read().strip()
    else:
        model_name = "Unknown"
    logger.info(f"Best model: {model_name}")

    model = joblib.load(MODELS_DIR / "best_model.joblib")

    # Load data
    X_test = pd.read_csv(DATA_PROCESSED / "X_test_scaled.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv")

    # Load feature names
    with open(MODELS_DIR / "selected_features.json", 'r') as f:
        feature_names = json.load(f)

    # Load label encoder
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")

    # Sample 500 instances for SHAP (faster)
    np.random.seed(RANDOM_SEED)
    sample_idx = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]
    y_sample = y_test.iloc[sample_idx]

    logger.info(f"Computing SHAP values for {len(X_sample)} samples...")

    # Create explainer based on model type
    is_tree = model_name in ['RandomForest', 'XGBoost', 'DecisionTree']

    if is_tree:
        explainer = shap.TreeExplainer(model)
    else:
        # For linear models, use a background dataset
        bg = shap.sample(X_test, 100)
        explainer = shap.LinearExplainer(model, bg)

    shap_values = explainer.shap_values(X_sample)

    # ── PLOT 1: Summary Plot (Beeswarm) ──
    logger.info("Creating SHAP summary plot...")
    try:
        fig = plt.figure(figsize=(12, 10))
        if isinstance(shap_values, list):
            # Multi-class: shap_values is a list of arrays (one per class)
            # Use mean absolute SHAP across classes
            shap_abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            shap.summary_plot(shap_abs_mean, X_sample, feature_names=feature_names,
                              show=False, max_display=20)
        else:
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                              show=False, max_display=20)
        plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: shap_summary.png")
    except Exception as e:
        logger.warning(f"  Summary plot failed: {e}")

    # ── PLOT 2: Feature Importance Bar Plot ──
    logger.info("Creating SHAP importance bar plot...")
    try:
        fig = plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        importance_df = pd.Series(mean_abs, index=feature_names).sort_values(ascending=True)

        plt.barh(range(len(importance_df)), importance_df.values,
                 color=plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df))))
        plt.yticks(range(len(importance_df)), importance_df.index, fontsize=9)
        plt.xlabel("Mean |SHAP Value|", fontsize=12)
        plt.title("SHAP Feature Importance", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_importance_bar.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: shap_importance_bar.png")
    except Exception as e:
        logger.warning(f"  Importance bar plot failed: {e}")

    # ── PLOT 3: Dependence Plots for Top 3 Features ──
    logger.info("Creating SHAP dependence plots for top 3 features...")
    try:
        if isinstance(shap_values, list):
            mean_imp = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
        else:
            mean_imp = np.abs(shap_values).mean(axis=0)
        top3_idx = np.argsort(mean_imp)[-3:][::-1]
        top3_names = [feature_names[i] for i in top3_idx]

        for feat_idx, feat_name in zip(top3_idx, top3_names):
            fig = plt.figure(figsize=(10, 6))
            sv = shap_values[0] if isinstance(shap_values, list) else shap_values
            shap.dependence_plot(feat_idx, sv, X_sample,
                                 feature_names=feature_names, show=False)
            plt.title(f"SHAP Dependence: {feat_name}", fontsize=13, fontweight='bold')
            plt.tight_layout()
            safe_name = feat_name.replace('/', '_').replace(' ', '_')
            plt.savefig(REPORTS_DIR / f"shap_dependence_{safe_name}.png",
                        dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved: shap_dependence_{safe_name}.png")
    except Exception as e:
        logger.warning(f"  Dependence plots failed: {e}")

    # ── PLOT 4: Individual Attack Explanations ──
    logger.info("Creating individual attack explanation plots...")
    try:
        y_pred = model.predict(X_sample)
        y_true = y_sample['Label_Multi'].values

        # Find correctly predicted attacks of different types
        attack_classes = [c for c in le.classes_ if c != 'BENIGN']
        explained = 0

        for cls_name in attack_classes:
            if explained >= 3:
                break
            cls_code = le.transform([cls_name])[0]
            mask = (y_true == cls_code) & (y_pred == cls_code)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            idx = indices[0]
            fig = plt.figure(figsize=(12, 6))

            if isinstance(shap_values, list):
                sv_single = shap_values[cls_code][idx]
            else:
                sv_single = shap_values[idx]

            # Bar plot of SHAP values for this instance
            sorted_idx = np.argsort(np.abs(sv_single))[-10:]
            plt.barh(range(len(sorted_idx)),
                     sv_single[sorted_idx],
                     color=['#e74c3c' if v > 0 else '#2ecc71' for v in sv_single[sorted_idx]])
            plt.yticks(range(len(sorted_idx)),
                       [feature_names[i] for i in sorted_idx], fontsize=10)
            plt.xlabel("SHAP Value (impact on prediction)", fontsize=11)
            plt.title(f"Why classified as '{cls_name}'? (Sample #{idx})",
                      fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / f"shap_explanation_sample_{explained+1}.png",
                        dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved: shap_explanation_sample_{explained+1}.png ({cls_name})")
            explained += 1

    except Exception as e:
        logger.warning(f"  Individual explanations failed: {e}")

    # ── Print Top 5 Features ──
    print("\n" + "=" * 60)
    print("TOP 5 MOST INFLUENTIAL FEATURES FOR DETECTING INTRUSIONS")
    print("=" * 60)
    if isinstance(shap_values, list):
        mean_imp = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
    else:
        mean_imp = np.abs(shap_values).mean(axis=0)

    top5_idx = np.argsort(mean_imp)[-5:][::-1]
    for rank, idx in enumerate(top5_idx, 1):
        print(f"  {rank}. {feature_names[idx]:40s} (mean |SHAP| = {mean_imp[idx]:.4f})")

    logger.info(f"\n✅ SHAP explainability complete. Plots saved to {REPORTS_DIR}")