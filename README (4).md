# 🛡️ Message Intelligence System

**Production-Grade Spam Classification using Distance-Based, Margin-Based, and Probabilistic Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)](#)

---

## 📌 Project Overview

The **Message Intelligence System** is an end-to-end machine learning pipeline built for a communication security company to automatically classify incoming digital messages as **Legitimate (0)** or **Spam (1)**.

The system combines **probability theory** with three complementary classification paradigms:

| Paradigm | Algorithm | Core Idea |
|---|---|---|
| **Distance-based** | K-Nearest Neighbors (KNN) | Classify by majority vote among the closest training examples |
| **Margin-based** | Support Vector Machine (SVM) | Find the hyperplane that maximizes separation between classes |
| **Probabilistic** | Naive Bayes (Gaussian) | Apply Bayes' Theorem under a conditional-independence assumption |

All three models are trained, tuned, and rigorously compared on a common, leakage-free preprocessing pipeline, with full **model artifact logging** for reproducibility and deployment.

### Key Engineering Principles Applied
- ✅ **Zero data leakage** — preprocessing (imputation + scaling) is fit exclusively on training data
- ✅ **Stratified splitting & cross-validation** — preserves the ~18.7% spam class ratio throughout
- ✅ **Hyperparameter tuning via GridSearchCV** — no hand-picked "magic numbers"
- ✅ **Multi-metric evaluation** — Accuracy, Precision, Recall, F1, and ROC-AUC (not accuracy alone, given class imbalance)
- ✅ **Versioned artifact logging** — every run persists models, preprocessor, and metrics as reproducible, timestamped artifacts
- ✅ **Deep inline documentation** — every functional code block explains the *engineering or statistical reason* behind the choice, not just what it does

---

## 🏗️ Architecture

```
┌──────────────────────┐
│   Raw CSV Dataset     │  message_text, length, urls, keyword scores,
│   (5,200 messages)    │  sender behavior, timestamp, spam_label
└───────────┬──────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│  Data Quality Audit                            │
│  • Missing value detection (3 columns)         │
│  • Duplicate check                             │
│  • Class balance analysis (~18.7% spam)         │
└───────────┬────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│  Preprocessing Pipeline (sklearn Pipeline +    │
│  ColumnTransformer)                             │
│  • SimpleImputer(strategy="median")             │
│  • StandardScaler (z-score normalization)       │
└───────────┬────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│  Stratified Train/Test Split (80/20)            │
└───────────┬────────────────────────────────────┘
            │
   ┌────────┼────────────────┬───────────────────┐
   ▼        ▼                ▼
┌──────┐ ┌──────┐        ┌────────────┐
│ KNN  │ │ SVM  │        │ Naive Bayes│
│(Grid │ │(Grid │        │ (Gaussian) │
│Search)│ │Search)│       │            │
└──┬───┘ └──┬───┘        └─────┬──────┘
   │        │                  │
   └────────┴─────────┬────────┘
                       ▼
        ┌───────────────────────────┐
        │  Unified Evaluation Layer  │
        │  Accuracy / Precision /    │
        │  Recall / F1 / ROC-AUC     │
        └─────────────┬─────────────┘
                       ▼
        ┌───────────────────────────┐
        │  Model Artifact Logging    │
        │  (joblib + JSON metadata,  │
        │   timestamped run folders) │
        └───────────────────────────┘
```

---

## 📂 Repository Structure

```
message-intelligence-system/
├── Message_Intelligence_System_Production.ipynb   # Full annotated notebook (Parts A–G)
├── Message_Intelligence_Dataset_5200.csv          # Source dataset
├── build_notebook.py                              # Script that generates the notebook programmatically
├── artifacts/                                     # Versioned model artifacts (generated per run)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── preprocessor.joblib
│       ├── model_knn.joblib
│       ├── model_svm.joblib
│       ├── model_naive_bayes.joblib
│       └── run_metadata.json
├── requirements.txt
└── README.md
```

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Handling | pandas, NumPy |
| Modeling | scikit-learn (KNeighborsClassifier, SVC, GaussianNB, GridSearchCV, Pipeline, ColumnTransformer) |
| Visualization | matplotlib, seaborn |
| Artifact Serialization | joblib, JSON |
| Notebook Environment | Jupyter / nbformat |

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/message-intelligence-system.git
cd message-intelligence-system

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook Message_Intelligence_System_Production.ipynb
```

### `requirements.txt`
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
nbformat>=5.9
jupyter>=1.0
```

---

## 🚀 Usage

1. **Run the notebook top-to-bottom** — each section is self-contained and clearly labeled (Part A through Part G + Appendix), matching the original assignment structure.
2. **Regenerate the notebook from scratch:**
   ```bash
   python build_notebook.py
   ```
   This is useful in CI pipelines to guarantee the notebook and its documentation never drift out of sync with the underlying code.
3. **Load a saved model for inference:**
   ```python
   import joblib

   preprocessor = joblib.load("artifacts/run_<timestamp>/preprocessor.joblib")
   model = joblib.load("artifacts/run_<timestamp>/model_svm.joblib")

   # new_data must be a DataFrame with the same 12 engineered feature columns
   X_new_processed = preprocessor.transform(new_data)
   predictions = model.predict(X_new_processed)
   ```

---

## 📊 Evaluation Methodology

Because spam represents only **~18.7%** of the dataset, this project deliberately avoids relying on Accuracy alone (a trivial "always predict legitimate" classifier would already score ~81%). Instead:

- **Stratified K-Fold Cross-Validation** (k=5) is used during hyperparameter search to ensure every fold reflects the true class ratio.
- **F1-score** is the primary model-selection metric during `GridSearchCV`, balancing Precision and Recall.
- **ROC-AUC** is reported as a threshold-independent measure of ranking quality, supporting flexible production threshold tuning.
- **Confusion matrices** are inspected per model to understand the real-world cost trade-off (false positives = blocked legitimate messages vs. false negatives = spam reaching the inbox).

---

## 🔮 Future MLOps Enhancements

| Enhancement | Description |
|---|---|
| **CI/CD Pipeline** | Automate notebook execution + metric regression checks (e.g., GitHub Actions) on every commit, failing the build if F1 drops below a defined threshold |
| **Model Registry** | Integrate with MLflow or a cloud model registry (SageMaker, Vertex AI) instead of local `artifacts/` folders for team-wide model versioning |
| **Drift Monitoring** | Add statistical drift detection (e.g., Evidently AI, population stability index) on incoming feature distributions to trigger automated retraining |
| **REST API Serving** | Wrap the best model + preprocessor in a FastAPI service with a `/predict` endpoint for real-time message scoring |
| **NLP Feature Expansion** | Incorporate TF-IDF or transformer-based embeddings directly from `message_text` to capture semantic spam signals beyond hand-engineered features |
| **A/B Testing Framework** | Systematically compare new model versions against the production incumbent on live traffic before full rollout |
| **Explainability Layer** | Add SHAP/LIME explanations per prediction for auditability and regulatory compliance in the security domain |
| **Automated Retraining** | Schedule periodic retraining on a rolling window of freshly labeled data to counter spam-pattern concept drift |
| **Data Validation Layer** | Add schema/contract validation (e.g., Great Expectations, Pandera) on incoming data before it enters the pipeline |

---

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.

---

## 🙌 Acknowledgements

Built as part of the **Message Intelligence System** coursework (Red & White Skill Education), extended into a production-grade MLOps reference implementation.
