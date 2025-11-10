# Credit Card Fraud Detection System
**Binary Classification with Imbalanced Data Handling**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/iKrish/SupervisedLearning/HEAD?labpath=fraud-detection-demo-a3.ipynb)

---

## Overview

This project implements a credit card fraud detection system using supervised machine learning to classify transactions as fraudulent or legitimate. The system addresses the challenge of extreme class imbalance (0.172% fraud rate) using SMOTE and compares three classification algorithms to identify the best performing model.

---

## AI Task Type

**Binary Classification** for fraud detection using supervised learning on highly imbalanced transaction data.

**Key Characteristics:**
- Learns from labeled historical transactions (fraudulent vs. legitimate)
- Handles extreme class imbalance (0.172% fraud rate)
- Multiple model comparison (Logistic Regression, Random Forest, XGBoost)
- SMOTE (Synthetic Minority Over-sampling) for training data balancing
- Focuses on recall optimization (catching fraud) while maintaining precision

---

## How to Run

### Option 1: Run in Browser (One-Click)
1. Click the **"launch binder"** badge at the top
2. Wait for environment to build (~4 minutes first time, includes downloading 150MB dataset)
3. Open `fraud-detection-demo-a3.ipynb`
4. Click **"Run"** → **"Run All Cells"**
5. Wait for execution to complete (~4-5 minutes)
6. Review results

**No installation required!** Runs directly in your browser via MyBinder.

**Note:** Total time for first run: ~8-9 minutes (4 min build + 4-5 min execution)

### Option 2: Run Locally

**Prerequisites:**
- Python 3.8+
- pip package manager

**Installation:**
```bash
# Clone the repository
git clone https://github.com/iKrish/SupervisedLearning.git
cd SupervisedLearning

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook fraud-detection-demo-a3.ipynb
```

**Execute:**
1. Open `fraud-detection-demo-a3.ipynb`
2. Click **"Kernel"** → **"Restart & Run All"**
3. Wait for execution to complete (~4-5 minutes)
4. Review results

---

## Project Structure

```
Supervised Learning/
│
├── fraud-detection-demo-a3.ipynb   # Main notebook (20 cells)
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore patterns
│
└── raw_data/                        # Dataset
    └── creditcard.csv               # 284,807 transactions
```

---

## Methodology

### 1. Data Preprocessing
- **Load dataset**: 284,807 credit card transactions
- **Feature scaling**: StandardScaler normalization (z-score)
- **Train/test split**: 80/20 stratified split (maintains 0.172% fraud rate in both sets)
- **Result**: 227,846 training samples, 56,961 test samples

### 2. Handle Class Imbalance
- **Problem**: Only 0.172% fraud (492 out of 284,807 transactions)
- **Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
- **SMOTE parameters**: 
  - `sampling_strategy=0.5` (fraud → 50% of legitimate, not 50/50 balance)
  - `k_neighbors=5` (interpolate between 5 nearest fraud neighbors)
- **Result**: Training data balanced to 1:2 ratio (113,726 fraud : 227,452 legitimate)
- **Critical**: SMOTE applied **only to training data**, test set remains original imbalanced distribution

### 3. Model Training
Train three classifiers on SMOTE-balanced training data:

**Logistic Regression (Baseline)**
- Linear classifier with sigmoid activation
- Fast training (~5-10 seconds)
- Interpretable coefficients
- Parameters: `max_iter=1000`, `solver='lbfgs'`

**Random Forest (Ensemble)**
- 100 decision trees voting in parallel
- Handles non-linear patterns
- Parameters: `n_estimators=100`, `max_depth=20`, `min_samples_split=10`

**XGBoost (Gradient Boosting)**
- 100 sequential trees, each correcting previous errors
- State-of-the-art performance
- Parameters: `learning_rate=0.1`, `max_depth=6`, `scale_pos_weight=10`
- `scale_pos_weight=10`: Extra emphasis on minority class (fraud)

### 4. Evaluation
Evaluate all models on **original imbalanced test set** (0.172% fraud rate):

**Metrics**:
- **Precision**: Of predicted frauds, how many are real? (TP / (TP + FP))
- **Recall**: Of actual frauds, how many did we catch? (TP / (TP + FN))
- **F1-Score**: Harmonic mean balancing precision and recall
- **ROC-AUC**: Overall ranking quality (0.5=random, 1.0=perfect)
- **Confusion Matrix**: TP, TN, FP, FN breakdown

**Visualizations**:
- Confusion matrix heatmaps (all 3 models)
- Metric comparison bar charts
- ROC curves (all models on one chart)

---

## Dataset

**Source**: Real European credit card transactions (September 2013)

- **Transactions**: 284,807 total
- **Frauds**: 492 (0.172% of all transactions)
- **Legitimate**: 284,315 (99.828% of all transactions)
- **Imbalance ratio**: 1:578 (fraud:legitimate)
- **Features**: 30 numerical features
  - `Time`: Seconds elapsed since first transaction
  - `V1-V28`: PCA-transformed features (anonymized for privacy)
  - `Amount`: Transaction amount in euros
  - `Class`: Target variable (0=legitimate, 1=fraud)

**Class Imbalance Challenge**:
A naive model predicting all transactions as legitimate would achieve 99.83% accuracy while detecting **zero fraud** - highlighting why accuracy is a misleading metric for imbalanced data.

---

## Key Results

**Model Performance** (evaluated on original imbalanced test set):

| Model | Precision | Recall | F1-Score | ROC-AUC | FP | FN |
|-------|-----------|--------|----------|---------|----|----|
| Logistic Regression | 0.11 | 0.90 | 0.19 | 0.970 | 741 | 10 |
| **Random Forest** | **0.75** | **0.85** | **0.79** | **0.978** | **28** | **15** |
| XGBoost | 0.17 | 0.87 | 0.28 | 0.979 | 414 | 13 |

**Winner**: Random Forest achieves the best balance with:
- **75% Precision**: Only 28 false alarms out of 111 fraud predictions
- **85% Recall**: Catches 83 out of 98 frauds
- **F1=0.79**: Best balance between precision and recall
- **ROC-AUC=0.978**: Excellent ranking quality

**Business Impact**:
- Catches 85% of fraudulent transactions (83 out of 98)
- Only 0.05% false positive rate (28 false alarms per 56,864 legitimate transactions)
- Misses only 15 frauds (vs. missing all 98 with naive approach)
- 99.92% overall accuracy

---

## Technical Highlights

**Handling Class Imbalance**:
- SMOTE generates plausible synthetic frauds via interpolation (not random duplication)
- Defense in depth: SMOTE (data level) + `scale_pos_weight` (algorithm level)
- Honest evaluation: Test set remains original 1:578 ratio

**Feature Engineering**:
- StandardScaler normalization: z = (x - mean) / std
- All features scaled to mean≈0, std≈1
- Critical for distance-based algorithms and gradient descent convergence

**Model Comparison**:
- Logistic Regression: Fast baseline, linear decision boundary
- Random Forest: Parallel ensemble, non-linear patterns
- XGBoost: Sequential boosting, error correction, state-of-the-art

**Evaluation Strategy**:
- Stratified train/test split maintains 0.172% fraud rate
- Focus on F1-Score (balances precision and recall)
- ROC-AUC for threshold-independent ranking quality
- Confusion matrix for error type analysis (FP vs. FN)

---

## Dependencies

**Core Libraries**:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML models, preprocessing, metrics
- `imbalanced-learn` - SMOTE implementation
- `xgboost` - Gradient boosting
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations

See `requirements.txt` for specific versions.


---

## Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Original research: Worldline and ULB Machine Learning Group
- SMOTE algorithm: Chawla et al. (2002)
