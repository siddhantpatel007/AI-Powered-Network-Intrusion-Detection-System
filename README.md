# 🛡️ AI-Powered Network Intrusion Detection System

An end-to-end machine learning pipeline that analyzes network traffic and classifies it as benign or malicious across 15 different attack categories, with an interactive Streamlit dashboard for real-time threat monitoring and model explainability.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Overview

Cyber attacks are growing in volume and complexity. Traditional rule-based intrusion detection systems struggle to keep up with evolving threats like DDoS, brute force, botnets, and port scanning. This project addresses that gap by applying machine learning to network traffic classification.

The system processes over 2.8 million network flow records from the CICIDS2017 benchmark dataset, engineers the most informative features, trains and compares multiple ML models, and provides an interactive dashboard for security analysts to monitor, classify, and investigate network threats in real time.

SHAP (SHapley Additive exPlanations) is integrated for model explainability — because in cybersecurity, knowing *why* traffic was flagged matters just as much as detecting it.

---

## ✨ Features

- **Data Pipeline** — Automated preprocessing of 2.8M+ network flow records (cleaning, encoding, scaling)
- **Feature Engineering** — Dimensionality reduction from 78 → 30 features using variance filtering, correlation analysis, and Random Forest importance
- **Multi-Model Comparison** — 4 ML models trained and evaluated: Random Forest, XGBoost, Decision Tree, Logistic Regression
- **Binary + Multi-Class Classification** — Detects benign vs attack traffic AND classifies 15 specific attack types
- **Hyperparameter Tuning** — RandomizedSearchCV optimization of top-performing models
- **Model Explainability** — SHAP analysis with summary, dependence, and individual prediction explanations
- **Interactive Dashboard** — 5-page Streamlit app with live prediction, model comparison, and severity-tiered alert monitoring

---

## 🎯 Attack Types Detected

| Category | Attack Types |
|----------|-------------|
| **DoS/DDoS** | DDoS, DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest |
| **Brute Force** | FTP-Patator, SSH-Patator |
| **Web Attacks** | Brute Force, XSS, SQL Injection |
| **Reconnaissance** | PortScan |
| **Other** | Bot, Infiltration, Heartbleed |

---

## 📊 Results

| Model | Task | Accuracy | F1 (Weighted) | F1 (Macro) | Training Time |
|-------|------|----------|---------------|------------|---------------|
| Random Forest | Multi-class | 0.987 | 0.987 | 0.891 | ~142s |
| XGBoost | Multi-class | 0.985 | 0.985 | 0.883 | ~99s |
| Decision Tree | Multi-class | 0.976 | 0.976 | 0.862 | ~12s |
| Logistic Regression | Multi-class | 0.871 | 0.869 | 0.542 | ~46s |

> *Update these numbers with your actual results from `reports/model_comparison.csv`*

---

## 🏗️ Project Structure

```
Capstone Project/
├── dashboard/
│   ├── __init__.py
│   └── app.py                    # Streamlit dashboard (5 pages)
├── Dataset/                       # Raw CICIDS2017 CSVs (not uploaded)
├── processed_data/                # Cleaned & split data (not uploaded)
├── models/                        # Trained models (not uploaded)
├── reports/                       # Charts & reports (not uploaded)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Project configuration & paths
│   ├── preprocessing.py           # Data loading & cleaning
│   ├── eda.py                     # Exploratory data analysis
│   ├── feature_engineering.py     # Feature selection pipeline
│   ├── data_split.py              # Train/val/test split + scaling
│   ├── train.py                   # Model training & evaluation
│   ├── hyperparameter_tuning.py   # Tuning top models
│   ├── visualize_results.py       # Comparison charts
│   └── explainability.py          # SHAP analysis
├── .gitignore
├── requirements.txt
├── run_all.bat                    # One-click pipeline runner
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11+ |
| **ML Models** | scikit-learn, XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Class Imbalance** | imbalanced-learn (SMOTE) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Explainability** | SHAP |
| **Dashboard** | Streamlit |
| **Model Persistence** | Joblib |

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.11+
- CICIDS2017 dataset ([Download from Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) or [UNB](https://www.unb.ca/cic/datasets/ids-2017.html))

### Setup

```bash
# Clone the repository
git clone https://github.com/siddhantpatel007/AI-Powered-Network-Intrusion-Detection-System.git
cd AI-Powered-Network-Intrusion-Detection-System

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset
Download the CICIDS2017 CSV files and place them in the `Dataset/` folder.

### Run the Pipeline

Run each script in order:

```bash
python src/config.py                   # Verify setup
python src/preprocessing.py            # Clean data (5-15 min)
python src/eda.py                      # Generate EDA charts
python src/feature_engineering.py      # Select top 30 features
python src/data_split.py               # Split + scale data
python src/train.py                    # Train 4 models (15-30 min)
python src/hyperparameter_tuning.py    # Tune best models (15-30 min)
python src/visualize_results.py        # Generate comparison charts
python src/explainability.py           # SHAP analysis
```

Or run everything at once (Windows):
```bash
run_all.bat
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```
Open http://localhost:8501 in your browser.

---

## 📈 Dashboard Pages

| Page | Description |
|------|-------------|
| **📊 Overview** | Dataset statistics, attack distribution, system architecture |
| **🤖 Model Comparison** | Side-by-side metrics, confusion matrices, ROC curves |
| **🔬 Feature Analysis** | Feature importance, correlation heatmap, SHAP plots |
| **🔍 Live Prediction** | Upload CSV or manually input features for real-time classification |
| **🚨 Alert Dashboard** | Severity-tiered alert monitoring with filters and downloadable reports |

---

## 📚 Dataset Citation

Sharafaldin, I., Lashkari, A.H., & Ghorbani, A.A. (2018). "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." 4th International Conference on Information Systems Security and Privacy (ICISSP).

---

## 🔮 Future Work

- Real-time packet capture integration using Scapy/PyShark
- Deep learning models (LSTM, 1D-CNN) for sequential pattern detection
- Adversarial robustness testing using IBM ART
- Multi-dataset validation (NSL-KDD, UNSW-NB15)
- Federated learning for privacy-preserving collaborative IDS
- Docker containerization for easy deployment

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.