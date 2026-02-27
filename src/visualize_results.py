"""
Phase 5.3 — Result Visualizations
Generates comparison charts: bar charts, ROC curves, per-class F1, time vs performance.

HOW TO RUN:
    cd "C:\Capstone Project"
    python src/visualize_results.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import sys, os
from sklearn.metrics import roc_curve, auc, f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')


def load_results():
    """Load model comparison CSV."""
    return pd.read_csv(REPORTS_DIR / "model_comparison.csv")


def plot_model_comparison(results: pd.DataFrame, task: str) -> None:
    """Grouped bar chart comparing all models on key metrics."""
    logger.info(f"Creating {task} model comparison chart...")
    df = results[results['task'] == task].copy()
    df = df.sort_values('f1_weighted', ascending=False)

    metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'recall_weighted']
    if task == 'binary' and 'roc_auc' in df.columns:
        metrics.append('roc_auc')
    metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]

    x = np.arange(len(df))
    width = 0.8 / len(metrics)
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, metric in enumerate(metrics):
        vals = df[metric].fillna(0).values
        bars = ax.bar(x + i * width, vals, width, label=metric.replace('_', ' ').title(), color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Performance Comparison — {task.title()} Classification',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * len(metrics) / 2)
    ax.set_xticklabels(df['model_name'], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = REPORTS_DIR / f"model_comparison_{task}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_roc_curves() -> None:
    """Plot ROC curves for all binary models on one figure."""
    logger.info("Creating ROC curves...")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test_scaled.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv")['Label_Binary']

    model_names = ["DecisionTree", "RandomForest", "XGBoost", "LogisticRegression"]
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.50)')

    for name, color in zip(model_names, colors):
        model_path = MODELS_DIR / f"{name}_binary.joblib"
        if not model_path.exists():
            logger.warning(f"  Model not found: {model_path}")
            continue

        model = joblib.load(model_path)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        except Exception as e:
            logger.warning(f"  ROC failed for {name}: {e}")

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Binary Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = REPORTS_DIR / "roc_curves_binary.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_per_class_f1() -> None:
    """Per-class F1 comparison across models."""
    logger.info("Creating per-class F1 comparison...")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test_scaled.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv")['Label_Multi']

    le_path = MODELS_DIR / "label_encoder.joblib"
    le = joblib.load(le_path) if le_path.exists() else None

    model_names = ["DecisionTree", "RandomForest", "XGBoost", "LogisticRegression"]
    all_f1s = {}

    classes_in_test = sorted(y_test.unique())
    class_names = [le.inverse_transform([c])[0] for c in classes_in_test] if le else [str(c) for c in classes_in_test]

    for name in model_names:
        model_path = MODELS_DIR / f"{name}_multi.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        f1s = f1_score(y_test, y_pred, labels=classes_in_test, average=None, zero_division=0)
        all_f1s[name] = f1s

    if not all_f1s:
        logger.warning("  No models found for per-class F1")
        return

    x = np.arange(len(class_names))
    width = 0.8 / len(all_f1s)
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_f1s)))

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (name, f1s) in enumerate(all_f1s.items()):
        ax.bar(x + i * width, f1s, width, label=name, color=colors[i])

    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1-Score by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * len(all_f1s) / 2)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = REPORTS_DIR / "per_class_f1_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_time_vs_performance(results: pd.DataFrame) -> None:
    """Scatter plot: training time vs F1 score."""
    logger.info("Creating time vs performance plot...")
    df = results[results['task'] == 'multi'].copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']

    for i, (_, row) in enumerate(df.iterrows()):
        c = colors[i % len(colors)]
        ax.scatter(row['training_time_sec'], row['f1_weighted'],
                   s=row['accuracy'] * 500, c=c, alpha=0.7, edgecolors='black', linewidth=1)
        ax.annotate(row['model_name'],
                    (row['training_time_sec'], row['f1_weighted']),
                    textcoords="offset points", xytext=(10, 10), fontsize=11, fontweight='bold')

    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('F1 Score (Weighted)', fontsize=12)
    ax.set_title('Training Time vs Performance Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = REPORTS_DIR / "time_vs_performance.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PHASE 5.3: Result Visualizations")
    logger.info("=" * 60)

    results = load_results()
    plot_model_comparison(results, 'multi')
    plot_model_comparison(results, 'binary')
    plot_roc_curves()
    plot_per_class_f1()
    plot_time_vs_performance(results)

    logger.info(f"\n✅ All visualizations saved to {REPORTS_DIR}")