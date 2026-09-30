# Telco Customer Churn: Classification Algorithm Benchmark

This repository benchmarks six standard machine learning classification models on the **IBM Telco Customer Churn** dataset to evaluate and compare predictive performance (Accuracy and F1 Score).

---

## Dataset Overview

* **Source**: [IBM Telco Customer Churn Dataset (GitHub Raw Link)](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
* **Target Variable**: `Churn` (Mapped: `Yes` -> 1, `No` -> 0)
* **Total Instances**: 7,032 samples (after removing 11 rows with missing `TotalCharges`)
* **Test Set Split**: 20% validation test set (N = 1,407 samples)
* **Features Used (4 Numerical Features)**:
  `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`
* **Preprocessing**: Coerced `TotalCharges` to numeric, dropped missing values, and scaled all features using `StandardScaler`.

---

## Algorithms Tested

1. **Logistic Regression**: Linear baseline classification model.
2. **Support Vector Classifier (`SVC`)**: Non-linear classifier utilizing an RBF kernel (C = 1.0).
3. **Decision Tree Classifier**: Non-parametric tree algorithm restricted to depth 5 to prevent overfitting.
4. **Random Forest Classifier**: Ensemble bagging technique combining 100 decision trees.
5. **AdaBoost Classifier**: Ensemble boosting technique adjusting sample weights sequentially across 100 trees.
6. **Gradient Boosting Classifier**: Ensemble boosting technique minimizing loss sequentially across 100 trees.

---

## Experimental Benchmark Results

Models ranked by overall **F1 Score** (critical metric for imbalanced churn classification) on the 20% validation test split:

| Rank | Model | Accuracy | F1 Score | Model Class |
| :-: | :--- | :-: | :-: | :--- |
| **1** | **Gradient Boosting Classifier** | **0.7854** | **0.5354** | `GradientBoostingClassifier` |
| **2** | **Decision Tree Classifier** | **0.7676** | **0.5170** | `DecisionTreeClassifier` |
| **3** | **Logistic Regression** | **0.7804** | **0.5134** | `LogisticRegression` |
| **4** | **AdaBoost Classifier** | **0.7761** | **0.4895** | `AdaBoostClassifier` |
| **5** | **Random Forest Classifier** | **0.7555** | **0.4835** | `RandomForestClassifier` |
| **6** | **Support Vector Classifier (SVC)** | **0.7825** | **0.4778** | `SVC` |

---

## Output Verification & Mathematical Calculations

To verify model evaluation consistency across the test set (N = 1,407 total test samples), we examine the mathematical formulation of **Accuracy** and **F1 Score**.

### Core Formulas

1. **Accuracy**:
   Accuracy = (True Positives + True Negatives) / Total Test Samples
   Total Correct Predictions = Accuracy * N_test

2. **Precision & Recall**:
   Precision = True Positives / (True Positives + False Positives)
   Recall = True Positives / (True Positives + False Negatives)

3. **F1 Score (Harmonic Mean of Precision & Recall)**:
   F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
            = 2 * True Positives / (2 * True Positives + False Positives + False Negatives)

---

### Detailed Calculations Breakdown

With a test size of N = 1,407 samples, we can compute the exact number of correct predictions made by each model:

* **Gradient Boosting Classifier:**
  * Correct Predictions = round( 0.7854 * 1407 ) = **1,105 / 1,407**
  * F1 Score = **0.5354** (Highest balance between capturing churners and minimizing false alarms)

* **Support Vector Classifier (SVC):**
  * Correct Predictions = round( 0.7825 * 1407 ) = **1,101 / 1,407**
  * F1 Score = **0.4778** (High overall accuracy, but poor minority class recall)

* **Logistic Regression:**
  * Correct Predictions = round( 0.7804 * 1407 ) = **1,098 / 1,407**
  * F1 Score = **0.5134**

* **AdaBoost Classifier:**
  * Correct Predictions = round( 0.7761 * 1407 ) = **1,092 / 1,407**
  * F1 Score = **0.4895**

* **Decision Tree Classifier:**
  * Correct Predictions = round( 0.7676 * 1407 ) = **1,080 / 1,407**
  * F1 Score = **0.5170**

* **Random Forest Classifier:**
  * Correct Predictions = round( 0.7555 * 1407 ) = **1,063 / 1,407**
  * F1 Score = **0.4835**

---

### Classification Metric Verification Table

| Model | Accuracy | Total Test Samples (N) | Implied Correct Predictions | F1 Score | Performance Status |
| :--- | :-: | :-: | :-: | :-: | :-: |
| **Gradient Boosting Classifier** | 0.7854 | 1,407 | **1,105** | **0.5354** | Best Overall |
| **SVC** | 0.7825 | 1,407 | **1,101** | **0.4778** | High Accuracy / Low Recall |
| **Logistic Regression** | 0.7804 | 1,407 | **1,098** | **0.5134** | Strong Linear Baseline |
| **AdaBoost Classifier** | 0.7761 | 1,407 | **1,092** | **0.4895** | Moderate |
| **Decision Tree Classifier** | 0.7676 | 1,407 | **1,080** | **0.5170** | Strong F1 Score |
| **Random Forest Classifier** | 0.7555 | 1,407 | **1,063** | **0.4835** | Overfitting Observed |

---

## Key Performance Insights

* **Top Performer**: **Gradient Boosting Classifier** achieved both the highest **Accuracy (78.54%)** and highest **F1 Score (0.5354)**, making it the most balanced model for detecting churning customers.
* **Accuracy vs. F1 Score Paradox**: While **SVC** achieved a high Accuracy (78.25%), its lower F1 Score (0.4778) indicates it predicts the majority non-churn class effectively but misses significant churning customers (high False Negatives).
* **Tree Depth Effect**: The constrained **Decision Tree (max_depth=5)** outperformed unconstrained ensemble models like Random Forest on F1 Score, showing that deeper tree ensembles overfit on this limited 4-feature subset.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone [https://github.com/your-username/telco-churn-benchmark.git](https://github.com/your-username/telco-churn-benchmark.git)
   cd telco-churn-benchmark