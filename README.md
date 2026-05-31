# 🤖 ChurnGuard AI v5.0 — Customer Churn Prediction & Retention Intelligence System

<div align="center">

### Machine Learning-Based Customer Churn Prediction System for the Telecommunications Industry

</div>

---

# 📋 Table of Contents

* [Overview](#-overview)
* [Features](#-features)
* [Best Model Performance](#-best-model-performance)
* [Models Comparison](#-models-comparison)
* [Tech Stack](#-tech-stack)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Pipeline](#-pipeline)
* [Customer Segments](#-customer-segments)
* [Authors](#-authors)

---

# 📌 Overview

ChurnGuard AI v5.0 is an end-to-end machine learning system developed to predict customer churn in the telecommunications industry using the IBM Telco Customer Churn dataset.

The system includes:

* Advanced data preprocessing
* Feature engineering
* Class imbalance handling using SMOTE
* 10 supervised machine learning classifiers
* 3 ensemble learning strategies
* 3 clustering algorithms
* Full hyperparameter tuning and evaluation

The final Tuned Stacking Ensemble v2 achieved approximately **82% test accuracy** with a **ROC-AUC score greater than 0.87**.

---

# ✨ Features

* Complete preprocessing pipeline
* Missing value handling
* Outlier treatment using Winsorization
* SMOTE-based imbalance handling
* Feature engineering
* 10 machine learning classifiers
* 3 ensemble learning methods
* 3 clustering algorithms
* Hyperparameter tuning using GridSearchCV, RandomizedSearchCV, and Optuna
* ROC curves and confusion matrix evaluation
* Streamlit deployment support
* Model serialization using Pickle

---

# 📊 Best Model Performance

## Tuned Stacking Ensemble v2

| Metric        | Result |
| ------------- | ------ |
| Test Accuracy | ~82%   |
| ROC-AUC       | > 0.87 |
| F1-Score      | ~0.62  |
| Precision     | ~0.96  |
| Recall        | ~0.99  |

---

# 🏆 Models Comparison

| Model               | Test Accuracy | F1 Score | ROC-AUC |
| ------------------- | ------------- | -------- | ------- |
| Logistic Regression | ~0.80         | ~0.59    | ~0.84   |
| Decision Tree       | ~0.77         | ~0.55    | ~0.71   |
| Random Forest       | ~0.80         | ~0.58    | ~0.85   |
| XGBoost             | ~0.81         | ~0.60    | ~0.86   |
| SVM (RBF)           | ~0.79         | ~0.57    | ~0.84   |
| k-NN                | ~0.77         | ~0.54    | ~0.78   |
| LightGBM            | ~0.81         | ~0.60    | ~0.86   |
| CatBoost            | ~0.81         | ~0.61    | ~0.87   |
| Gradient Boosting   | ~0.80         | ~0.59    | ~0.85   |
| ExtraTrees          | ~0.79         | ~0.57    | ~0.84   |
| Stacking Ensemble   | ~0.81         | ~0.61    | ~0.87   |
| Voting Ensemble     | ~0.81         | ~0.60    | ~0.87   |
| Weighted Ensemble   | ~0.81         | ~0.60    | ~0.87   |
| Tuned Stacking v2 ⭐ | ~0.82         | ~0.62    | ~0.87   |

---

# 🛠️ Tech Stack

| Category              | Technologies                              |
| --------------------- | ----------------------------------------- |
| Language              | Python 3                                  |
| ML Libraries          | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Data Processing       | Pandas, NumPy                             |
| Visualization         | Matplotlib, Seaborn                       |
| Imbalance Handling    | SMOTE                                     |
| Hyperparameter Tuning | GridSearchCV, RandomizedSearchCV, Optuna  |
| Deployment            | Streamlit, Pickle                         |
| Environment           | Jupyter Notebook                          |

---

# 📁 Project Structure

```text id="hjcnj4"
churnguard_ai/
│
├── Churn_Prediction_FINAL_v5.ipynb
├── app.py
├── churn_model.pkl
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Install Dependencies

```bash id="m83a8m"
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn matplotlib seaborn optuna streamlit
```

---

# 🚀 How to Run

## Run Jupyter Notebook

```bash id="crmd3y"
jupyter notebook Churn_Prediction_FINAL_v5.ipynb
```

## Run Streamlit App

```bash id="hgjv7q"
streamlit run app.py
```

---

# 🔄 Machine Learning Pipeline

## Phase 1 — Data Preprocessing

* Missing value handling
* Type conversion fixes
* Winsorization for outlier treatment
* Feature engineering
* Label Encoding and One-Hot Encoding
* SMOTE applied only on training data
* Stratified train/test split

### Engineered Features

* TotalServices
* AvgMonthlySpend
* ContractTypeScore
* Fiber_NoSecurity
* Fiber_NoTech

---

## Phase 2 — Model Implementation

### Supervised Models

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Gradient Boosting
* Support Vector Machine
* k-Nearest Neighbors
* ExtraTrees

### Ensemble Methods

* Stacking Ensemble
* Voting Ensemble
* Weighted Ensemble

### Clustering Algorithms

* K-Means
* Agglomerative Clustering
* DBSCAN

---

## Phase 3 — Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Additional evaluation:

* ROC Curves
* Confusion Matrices
* Feature Importance Analysis
* Cluster Profiling

---

## Phase 4 — Hyperparameter Tuning

### Techniques Used

* GridSearchCV
* RandomizedSearchCV
* Bayesian Optimization using Optuna

### Tuned Models

* Random Forest
* LightGBM
* Tuned Stacking Ensemble v2

---

# 👥 Customer Segments

| Cluster   | Average Tenure | Churn Rate | Risk Level |
| --------- | -------------- | ---------- | ---------- |
| Cluster 0 | ~7 months      | ~55%       | High       |
| Cluster 1 | ~24 months     | ~25%       | Medium     |
| Cluster 2 | ~55 months     | ~7%        | Low        |

### Insights

* Cluster 0 contains high-risk customers with the highest churn rate.
* Cluster 1 represents medium-risk customers suitable for retention offers.
* Cluster 2 contains loyal long-term customers with low churn probability.

---

# 👩‍💻 Authors

| Name           | Role                                |
| -------------- | ----------------------------------- |
| Sabahat Fatima | Developer                           |
| Project Team   | Machine Learning System Development |

### Institution

Fatima Jinnah Women University
Department of Software Engineering

### Course Information

Machine Learning (SE-CD-638)
Semester VI

### Instructor

Dr. Aamir Arsalan

---

<div align="center">

### Built using Machine Learning & Artificial Intelligence

</div>
