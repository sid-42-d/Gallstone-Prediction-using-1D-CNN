# Gallstone-Prediction-CNN 🏥🔬

A deep learning-based binary classification project to predict gallstone presence using bioimpedance measurements and laboratory biomarkers with 1D Convolutional Neural Networks (CNNs) and interpretable AI techniques (SHAP/LIME).

---

## 📌 Project Overview

This project predicts gallstone presence using clinical measurements into two categories:

* **Gallstone Present** (Class 0)
* **No Gallstone** (Class 1)

Goal: Assist medical professionals in early gallstone detection using non-invasive bioimpedance analysis and routine laboratory tests with interpretable AI.

---

## 🚀 Model Highlights

* **Base Model:** Custom 1D CNN for tabular medical data
* **Architecture:** Conv1D (64→32) + Dense (96→16) + Dropout (0.3)
* **Optimization:** Hyperband algorithm (~90 trials)
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam (lr=0.001)
* **Interpretability:** SHAP + LIME for clinical validation

---

## 📂 Dataset

* **Source:** [UCI Machine Learning Repository - Gallstone Dataset](https://archive.ics.uci.edu/dataset/948/gallstone)
* **Total Samples:** 319 patients
* **Features:** 38 bioimpedance + lab measurements

| Split | Gallstone | No Gallstone | Total |
|-------|-----------|--------------|-------|
| Training | 96 | 95 | 191 |
| Validation | 24 | 24 | 48 |
| Test | 41 | 39 | 80 |

---

## 🧪 Feature Engineering

Created 6 clinically meaningful features:

* BMI_Risk (Obesity indicator)
* VitD_Deficient (< 20 ng/mL)
* High_CRP (> 3 mg/L)
* Fat_Lean_Ratio
* TC_HDL_Ratio
* AST_ALT_Ratio

**Result:** 38 → 44 features

---

## 🧠 Model Architecture
```python
Sequential([
    Conv1D(64, 3, activation='relu'),
    BatchNormalization(),
    Conv1D(32, 2, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(96, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

**Optimal Hyperparameters (via Hyperband):**

| Parameter | Value |
|-----------|-------|
| Conv1 Filters | 64 |
| Conv2 Filters | 32 |
| Dense1 Units | 96 |
| Dense2 Units | 16 |
| Dropout | 0.3 |
| Learning Rate | 0.001 |

---

## 📊 Results

### ✅ Single Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **85.00%** |
| Precision | 86% |
| Recall | 85% |
| F1-Score | 85% |
| ROC-AUC | 0.91 |

**Confusion Matrix:**
```
               Predicted
             Gallstone  No Gallstone
Gallstone        31          9
No Gallstone      3         37

Sensitivity: 78%  |  Specificity: 93%
```

### 📈 Advanced Methods

| Model | Accuracy |
|-------|----------|
| Baseline CNN | 82.50% |
| Optimized CNN | 85.00% |
| Ensemble (5 models) | 86.25% |
| Stacking (CNN + GB) | **87.50%** |

### 🎯 Comparison with State-of-the-Art

| Study | Method | Accuracy |
|-------|--------|----------|
| Esen et al. (2024) | Gradient Boosting | 85.42% |
| **Our Work (CNN)** | 1D CNN | **85.00%** |
| **Our Work (Ensemble)** | Stacking | **87.50%** |

---

## 🔍 SHAP Interpretability

**Top 10 Features:**

| Rank | Feature | SHAP Score |
|------|---------|------------|
| 1 | Vitamin D | 0.0847 |
| 2 | CRP | 0.0765 |
| 3 | Total Body Water | 0.0692 |
| 4 | BMI | 0.0634 |
| 5 | Lean Mass | 0.0589 |
| 6 | Cholesterol Ratio | 0.0523 |
| 7 | Age | 0.0498 |
| 8 | Triglyceride | 0.0456 |
| 9 | Extracellular Water | 0.0421 |
| 10 | Total Cholesterol | 0.0387 |

**Validation:** 75% agreement with original study (3/4 top features match)

---

---

## 💻 Technologies Used

* Python 3.8+
* TensorFlow 2.19
* Keras Tuner 1.3
* SHAP 0.41, LIME 0.2
* scikit-learn 1.6
* NumPy, Pandas
* Matplotlib, Seaborn

---
