"""
Phase 7 — Streamlit Dashboard (5 Pages)
Interactive dashboard for the AI-Powered NIDS.

HOW TO RUN:
    cd "C:\Capstone Project"
    pip install Pillow  (one time only)
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image
import sys, os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI-NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# IMAGE HELPER — FIXED VERSION
# ============================================================
def show_image(filename, caption=None, width=None):
    """
    Load and display an image from the reports folder.
    Shows info message if file not found.
    """
    path = REPORTS_DIR / filename
    if path.exists():
        try:
            img = Image.open(path)
            if width:
                st.image(img, caption=caption, width=width)
            else:
                st.image(img, caption=caption, use_column_width=True)
        except Exception as e:
            # Fallback: read as bytes
            try:
                with open(path, 'rb') as f:
                    img_bytes = f.read()
                if width:
                    st.image(img_bytes, caption=caption, width=width)
                else:
                    st.image(img_bytes, caption=caption, use_column_width=True)
            except Exception as e2:
                st.warning(f"Could not load image {filename}: {e2}")
    else:
        st.info(f"📁 Image not found: `{filename}`. Run the corresponding script first.")


# ============================================================
# CACHED DATA LOADING
# ============================================================
@st.cache_data
def load_cleaned_data():
    try:
        return pd.read_csv(DATA_PROCESSED / "cicids2017_cleaned.csv", low_memory=False)
    except Exception:
        return None

@st.cache_data
def load_model_comparison():
    try:
        return pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    except Exception:
        return None

@st.cache_data
def load_test_data():
    try:
        X = pd.read_csv(DATA_PROCESSED / "X_test_scaled.csv")
        y = pd.read_csv(DATA_PROCESSED / "y_test.csv")
        return X, y
    except Exception:
        return None, None

@st.cache_resource
def load_best_model():
    try:
        return joblib.load(MODELS_DIR / "best_model.joblib")
    except Exception:
        return None

@st.cache_resource
def load_scaler():
    try:
        return joblib.load(MODELS_DIR / "scaler.joblib")
    except Exception:
        return None

@st.cache_resource
def load_label_encoder():
    try:
        return joblib.load(MODELS_DIR / "label_encoder.joblib")
    except Exception:
        return None

@st.cache_data
def load_features():
    try:
        with open(MODELS_DIR / "selected_features.json", 'r') as f:
            return json.load(f)
    except Exception:
        return None


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("# 🛡️ AI-NIDS")
    st.markdown("**v1.0** | Network Intrusion Detection")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🤖 Model Comparison", "🔬 Feature Analysis",
         "🔍 Live Prediction", "🚨 Alert Dashboard"],
        index=0
    )

    st.markdown("---")
    st.markdown(
        "<small>AI-Powered NIDS"
        "</small>",
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "📊 Overview":
    st.title("🛡️ AI-Powered Network Intrusion Detection System")
    st.markdown("---")

    df = load_cleaned_data()
    results = load_model_comparison()
    features = load_features()

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total = f"{len(df):,}" if df is not None else "N/A"
        st.metric("Total Samples", total)
    with c2:
        n_feat = str(len(features)) if features else "N/A"
        st.metric("Features Selected", n_feat)
    with c3:
        n_attacks = str(df['Label'].nunique()) if df is not None else "N/A"
        st.metric("Attack Types", n_attacks)
    with c4:
        if results is not None:
            best_f1 = results[results['task'] == 'multi']['f1_weighted'].max()
            st.metric("Best Model F1", f"{best_f1:.4f}")
        else:
            st.metric("Best Model F1", "N/A")

    st.markdown("---")

    if df is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Attack Type Distribution")
            dist = df['Label'].value_counts().reset_index()
            dist.columns = ['Attack Type', 'Count']
            fig = px.bar(dist, x='Count', y='Attack Type', orientation='h',
                         color='Count', color_continuous_scale='Reds')
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Benign vs Attack Ratio")
            binary_dist = df['Label_Binary'].value_counts().reset_index()
            binary_dist.columns = ['Type', 'Count']
            binary_dist['Type'] = binary_dist['Type'].map({0: 'Benign', 1: 'Attack'})
            fig = px.pie(binary_dist, values='Count', names='Type',
                         color='Type', color_discrete_map={'Benign': '#2ecc71', 'Attack': '#e74c3c'})
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Dataset Sample (first 20 rows)"):
            st.dataframe(df.head(20), use_container_width=True)

        # System Architecture
        st.markdown("---")
        st.subheader("System Architecture")
        st.markdown("""
        ```
        ┌──────────┐    ┌───────────────┐    ┌─────────────┐    ┌──────────┐    ┌────────────┐    ┌───────────┐
        │ Raw Data │ →  │ Preprocessing │ →  │  Feature    │ →  │  Model   │ →  │ Prediction │ →  │ Dashboard │
        │ (8 CSVs) │    │ Clean & Encode│    │  Selection  │    │ Training │    │  & Alerts  │    │   (UI)    │
        └──────────┘    └───────────────┘    └─────────────┘    └──────────┘    └────────────┘    └───────────┘
              ↓                  ↓                  ↓                 ↓                ↓
          2.8M rows       Remove NaN/Inf       78 → 30          4 Models +       SHAP + Live
                          Strip columns        features         Hypertuning       Prediction
        ```
        """)
    else:
        st.warning("⚠️ Cleaned data not found. Run `python src/preprocessing.py` first.")


# ============================================================
# PAGE 2: MODEL COMPARISON
# ============================================================
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    st.markdown("---")

    results = load_model_comparison()

    if results is not None:
        task = st.selectbox("Select Classification Task", ["multi", "binary"])
        task_df = results[results['task'] == task].sort_values('f1_weighted', ascending=False)

        # Metrics table
        st.subheader("Performance Metrics")
        display_cols = ['model_name', 'accuracy', 'f1_weighted', 'f1_macro',
                        'precision_weighted', 'recall_weighted', 'roc_auc', 'training_time_sec']
        display_cols = [c for c in display_cols if c in task_df.columns]
        st.dataframe(task_df[display_cols].reset_index(drop=True), use_container_width=True)

        # Grouped bar chart
        st.subheader("Visual Comparison")
        metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'recall_weighted']
        metrics = [m for m in metrics if m in task_df.columns]

        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=task_df['model_name'],
                y=task_df[metric],
                text=task_df[metric].round(3),
                textposition='auto'
            ))
        fig.update_layout(barmode='group', height=500,
                          title=f"Model Comparison — {task.title()} Classification",
                          yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

        # Individual model details
        st.markdown("---")
        st.subheader("Detailed Model Report")
        model_choice = st.selectbox("Select Model", task_df['model_name'].tolist())

        # Show confusion matrix
        show_image(f"{model_choice}_{task}_cm.png",
                   caption=f"{model_choice} — {task.title()} Confusion Matrix")

        # Show classification report
        report_path = REPORTS_DIR / f"{model_choice}_{task}_report.txt"
        if report_path.exists():
            with open(report_path, 'r') as f:
                st.text(f.read())
        else:
            st.info(f"Report not found: {model_choice}_{task}_report.txt")

        # ROC Curves for binary
        if task == 'binary':
            st.subheader("ROC Curves")
            show_image("roc_curves_binary.png")

        # Download
        csv = task_df.to_csv(index=False)
        st.download_button("📥 Download Results CSV", csv,
                           f"model_comparison_{task}.csv", "text/csv")
    else:
        st.warning("⚠️ Model comparison results not found. Run `python src/train.py` first.")


# ============================================================
# PAGE 3: FEATURE ANALYSIS
# ============================================================
elif page == "🔬 Feature Analysis":
    st.title("🔬 Feature Analysis")
    st.markdown("---")

    # Feature Importance
    st.subheader("Random Forest Feature Importance")
    show_image("feature_importance_rf.png")

    st.markdown("---")

    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    show_image("correlation_heatmap.png")

    st.markdown("---")
    st.subheader("SHAP Analysis")

    col1, col2 = st.columns(2)
    with col1:
        show_image("shap_summary.png", caption="SHAP Summary")
    with col2:
        show_image("shap_importance_bar.png", caption="SHAP Feature Importance")

    st.markdown("---")

    # SHAP Dependence Plots
    st.subheader("SHAP Dependence Plots")
    dep_files = list(REPORTS_DIR.glob("shap_dependence_*.png"))
    if dep_files:
        cols = st.columns(min(3, len(dep_files)))
        for i, dep_file in enumerate(dep_files[:3]):
            with cols[i]:
                show_image(dep_file.name, caption=dep_file.stem.replace("shap_dependence_", ""))
    else:
        st.info("No SHAP dependence plots found. Run `python src/explainability.py` first.")

    # Individual Explanations
    with st.expander("🔍 Individual Attack Explanations"):
        found_any = False
        for i in range(1, 4):
            fname = f"shap_explanation_sample_{i}.png"
            if (REPORTS_DIR / fname).exists():
                show_image(fname, caption=f"Attack Sample {i}")
                st.markdown("---")
                found_any = True
        if not found_any:
            st.info("No individual explanations found. Run `python src/explainability.py` first.")


# ============================================================
# PAGE 4: LIVE PREDICTION
# ============================================================
elif page == "🔍 Live Prediction":
    st.title("🔍 Predict Network Traffic")
    st.markdown("---")

    model = load_best_model()
    scaler = load_scaler()
    le = load_label_encoder()
    features = load_features()

    if model is None or scaler is None or features is None:
        st.error("⚠️ Model, scaler, or features not found. Run all training scripts first.")
    else:
        mode = st.radio("Input Mode", ["📁 Upload CSV", "✏️ Manual Input"], horizontal=True)

        if mode == "📁 Upload CSV":
            uploaded = st.file_uploader("Upload a CSV file with network flow features", type=['csv'])
            if uploaded:
                with st.spinner("Processing..."):
                    input_df = pd.read_csv(uploaded)
                    input_df.columns = input_df.columns.str.strip()

                    # Check if required features exist
                    missing = [f for f in features if f not in input_df.columns]
                    if missing:
                        st.error(f"Missing features in uploaded CSV: {missing[:10]}...")
                        st.info(f"Your CSV needs these columns: {features}")
                    else:
                        X_input = input_df[features]
                        X_scaled = scaler.transform(X_input)
                        preds = model.predict(X_scaled)
                        proba = model.predict_proba(X_scaled)
                        confidence = np.max(proba, axis=1)

                        if le:
                            pred_labels = le.inverse_transform(preds)
                        else:
                            pred_labels = preds

                        result_df = input_df.copy()
                        result_df['Predicted'] = pred_labels
                        result_df['Confidence'] = (confidence * 100).round(1)
                        result_df['Status'] = ['🟢 Benign' if p == 'BENIGN' else '🔴 Attack'
                                                for p in pred_labels]

                        # Summary
                        n_benign = (pred_labels == 'BENIGN').sum()
                        n_attack = len(pred_labels) - n_benign
                        c1, c2 = st.columns(2)
                        c1.metric("Benign Traffic", f"{n_benign:,}")
                        c2.metric("Attacks Detected", f"{n_attack:,}")

                        st.dataframe(
                            result_df[['Status', 'Predicted', 'Confidence']].head(500),
                            use_container_width=True
                        )

                        csv = result_df.to_csv(index=False)
                        st.download_button("📥 Download Predictions", csv,
                                           "predictions.csv", "text/csv")

        else:  # Manual Input
            st.markdown("Enter values for key network flow features:")
            cols = st.columns(2)
            input_values = {}

            # Load training data to get median defaults
            try:
                X_train = pd.read_csv(DATA_PROCESSED / "X_train_scaled.csv", nrows=100)
                defaults = X_train.median().to_dict()
            except Exception:
                defaults = {f: 0.0 for f in features}

            for i, feat in enumerate(features[:10]):
                with cols[i % 2]:
                    default = defaults.get(feat, 0.0)
                    input_values[feat] = st.number_input(
                        feat, value=float(default), format="%.4f", key=feat
                    )

            # Fill remaining features with defaults
            for feat in features[10:]:
                input_values[feat] = defaults.get(feat, 0.0)

            if st.button("🔍 Predict", type="primary"):
                with st.spinner("Analyzing network traffic..."):
                    X_input = pd.DataFrame([input_values])[features]
                    pred = model.predict(X_input)[0]
                    proba = model.predict_proba(X_input)[0]
                    confidence = np.max(proba) * 100

                    if le:
                        pred_label = le.inverse_transform([pred])[0]
                    else:
                        pred_label = str(pred)

                    st.markdown("---")
                    if pred_label == 'BENIGN':
                        st.success(f"✅ **Prediction: BENIGN** (Confidence: {confidence:.1f}%)")
                    else:
                        st.error(f"⚠️ **Prediction: {pred_label}** (Confidence: {confidence:.1f}%)")

                    # Confidence chart — top 5 classes
                    if le:
                        prob_df = pd.DataFrame({
                            'Class': le.classes_,
                            'Probability (%)': (proba * 100).round(2)
                        }).sort_values('Probability (%)', ascending=True).tail(5)
                        fig = px.barh(prob_df, x='Probability (%)', y='Class',
                                      color='Probability (%)', color_continuous_scale='RdYlGn_r')
                        fig.update_layout(height=300, title="Top 5 Prediction Probabilities")
                        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 5: ALERT DASHBOARD
# ============================================================
elif page == "🚨 Alert Dashboard":
    st.title("🚨 Intrusion Alert Dashboard")
    st.markdown("---")

    model = load_best_model()
    le = load_label_encoder()
    X_test, y_test = load_test_data()

    if model is None or X_test is None:
        st.error("⚠️ Model or test data not found. Run all training scripts first.")
    else:
        # Predict on test set
        with st.spinner("Running intrusion detection on test data..."):
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)
            confidence = np.max(proba, axis=1)

            if le:
                pred_labels = le.inverse_transform(preds)
            else:
                pred_labels = preds.astype(str)

        # Filter only attacks
        attack_mask = pred_labels != 'BENIGN'
        attack_labels = pred_labels[attack_mask]
        attack_conf = confidence[attack_mask]

        # Severity mapping
        severity_map = {
            'DDoS': 'High', 'DoS Hulk': 'High', 'DoS GoldenEye': 'High', 'Bot': 'High',
            'PortScan': 'Medium', 'FTP-Patator': 'Medium', 'SSH-Patator': 'Medium',
        }

        severities = [severity_map.get(a, 'Low') for a in attack_labels]

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚨 Total Alerts", f"{len(attack_labels):,}")
        c2.metric("🔴 High Severity", f"{severities.count('High'):,}")
        c3.metric("🟠 Medium Severity", f"{severities.count('Medium'):,}")
        c4.metric("🟡 Low Severity", f"{severities.count('Low'):,}")

        st.markdown("---")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            sev_filter = st.multiselect("Filter by Severity",
                                        ['High', 'Medium', 'Low'],
                                        default=['High', 'Medium', 'Low'])
        with col2:
            attack_types = sorted(set(attack_labels))
            type_filter = st.multiselect("Filter by Attack Type",
                                         attack_types, default=attack_types)

        # Build alerts DataFrame
        base_time = datetime.now() - timedelta(hours=len(attack_labels) // 100)
        timestamps = [base_time + timedelta(seconds=i * 3) for i in range(len(attack_labels))]

        alerts_df = pd.DataFrame({
            'Timestamp': timestamps[:len(attack_labels)],
            'Attack Type': attack_labels,
            'Confidence (%)': (attack_conf * 100).round(1),
            'Severity': severities,
        })
        alerts_df['Alert ID'] = [f"ALT-{i+1:05d}" for i in range(len(alerts_df))]

        # Apply filters
        filtered = alerts_df[
            (alerts_df['Severity'].isin(sev_filter)) &
            (alerts_df['Attack Type'].isin(type_filter))
        ]

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            type_counts = filtered['Attack Type'].value_counts().reset_index()
            type_counts.columns = ['Attack Type', 'Count']
            fig = px.bar(type_counts, x='Attack Type', y='Count', color='Count',
                         color_continuous_scale='Reds', title="Alerts by Attack Type")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sev_counts = filtered['Severity'].value_counts().reset_index()
            sev_counts.columns = ['Severity', 'Count']
            fig = px.pie(sev_counts, values='Count', names='Severity',
                         color='Severity',
                         color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#f1c40f'},
                         title="Alerts by Severity")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Alert table (show latest 500)
        st.subheader(f"Alert Feed ({len(filtered):,} alerts)")
        display_df = filtered[['Alert ID', 'Timestamp', 'Attack Type',
                                'Confidence (%)', 'Severity']].head(500)
        st.dataframe(display_df, use_container_width=True, height=400)

        # Download
        csv = filtered.to_csv(index=False)
        st.download_button("📥 Download Alerts CSV", csv, "alerts.csv", "text/csv")