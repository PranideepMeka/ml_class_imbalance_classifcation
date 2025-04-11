# ml_class_imbalance_classifcation
Different Classifcaton algorithems and SMOTE analysis of the oil spill data set 
The Oil Spill dataset comes from the UCI Machine Learning Repository, a well-known and trusted source for datasets used in machine learning and data science research.

🔍 Dataset Source:
Name: Oil Spill Detection Dataset

Repository: UCI Machine Learning Repository

Original Link: https://archive.ics.uci.edu/ml/datasets/Oil+Spill

📄 Description:
This dataset was originally used for SAR (Synthetic Aperture Radar) image analysis, to detect oil spills in the ocean. Each row in the dataset represents a segment extracted from SAR images. The attributes describe statistical and physical properties of the image segments.

Instances: 937

Features: 48 numerical attributes (like texture, contrast, etc.)

Target: 1 = Oil Spill, 0 = Look-alike (non-spill, such as ocean wave or ship)

---

# 🛢️ Oil Spill Dataset: Analysis & Classification Modeling Report

## 📊 Dataset Overview

The dataset comprises **937 rows and 48 columns**, with the objective of predicting oil spills. The **target variable (`Target`)** is binary (0 or 1), indicating whether an oil spill occurred. The dataset includes a mix of numerical and categorical variables.

### ✅ Key Characteristics:
- **Target Type**: Binary classification (0 or 1)
- **No Missing Values**: Dataset is complete
- **Class Imbalance**: Present — addressed with **SMOTE** (Synthetic Minority Oversampling Technique)

---

## 🧹 Data Preprocessing

1. **Column Renaming**: All columns were renamed to `col_0` through `col_49` for clarity.
2. **Target Separation**: The `Target` column was isolated from the features.
3. **Feature Scaling**: All features were scaled using **StandardScaler**.
4. **Imbalance Correction**: SMOTE was applied to balance the class distribution and improve model performance.

---

## 📈 Exploratory Data Analysis (EDA)

### 🧮 Basic Insights:
- **Descriptive Stats**: Summary stats via `df.describe()` and structure using `df.info()`
- **Duplicates Check**: Identified and counted
- **Unique Values**: Counted across each column

### 📊 Visualizations:
- **Count Plot**: Distribution of the target classes
- **Histograms**: Distribution of individual features
- **Violin Plots**: Feature distributions with respect to the target
- **Correlation Heatmap**: Relationships between features

---

## 🤖 Machine Learning Models

A variety of classification models were applied, each evaluated for accuracy. Both default and custom hyperparameters were tested.

### 1. **Logistic Regression**
Tested with four solvers:
- `liblinear` → ⭐ **Best: 97.52%**
- `newton-cg`, `newton-cholesky`, `saga`

### 2. **K-Nearest Neighbors (KNN)**
- Configurations tested with different neighbors and algorithms:
  - Best: **89.36%** using `n_neighbors=10`, `algorithm='brute'`

### 3. **Decision Tree**
- Two splits evaluated:
  - Best: **94.68%** using `criterion='gini'`

### 4. **Random Forest**
- Estimators and criteria varied:
  - Best: **95.74%** with `n_estimators=150`, `criterion='entropy'`

---

## 📊 Model Comparison Summary

| Model                        | Hyperparameters                                                              | Accuracy Score | SMOTE Used |
|-----------------------------|------------------------------------------------------------------------------|----------------|------------|
| **Logistic Regression (lr1)** | `penalty='l2', solver='liblinear'`                                           | **97.52%**     | ❌ No       |
| Logistic Regression (lr2)   | `penalty='l2', solver='newton-cg'`                                          | 97.16%         | ❌ No       |
| Logistic Regression (lr3)   | `penalty='l2', solver='newton-cholesky'`                                    | 97.16%         | ❌ No       |
| Logistic Regression (lr4)   | `penalty='elasticnet', solver='saga', l1_ratio=0.5`                          | 96.81%         | ❌ No       |
| KNN (knn1)                  | `n_neighbors=30, algorithm='ball_tree'`                                     | 77.66%         | ✅ Yes      |
| KNN (knn2)                  | `n_neighbors=10, algorithm='brute'`                                         | 89.36%         | ✅ Yes      |
| Decision Tree (dtc1)        | `criterion='entropy', splitter='random'`                                    | 91.84%         | ✅ Yes      |
| **Decision Tree (dtc2)**     | `criterion='gini', splitter='best'`                                         | **94.68%**     | ✅ Yes      |
| **Random Forest (rfc_1)**    | `n_estimators=150, criterion='entropy'`                                     | **95.74%**     | ✅ Yes      |
| Random Forest (rfc_2)       | `n_estimators=100, criterion='gini'`                                        | 95.39%         | ✅ Yes      |

---

## 🧾 Conclusion

- **Logistic Regression (liblinear)** emerged as the top performer, achieving **97.52% accuracy without SMOTE**.
- **Random Forest** and **Decision Tree** showed strong results with **SMOTE applied**, at **95.74%** and **94.68%**, respectively.
- SMOTE was **effective** in improving the performance of models sensitive to class imbalance like **KNN** and **Decision Trees**.

---

## 💡 Recommendations

1. **Feature Selection**: Perform further analysis to identify redundant or low-impact features.
2. **Hyperparameter Optimization**: Use **GridSearchCV** or **RandomizedSearchCV** for fine-tuning.
3. **Ensemble Approaches**: Consider advanced ensemble techniques like **Stacking**, **Voting**, or **Boosting** for potentially higher accuracy.
4. **Model Explainability**: Apply SHAP or LIME to understand feature contributions for more transparent decisions.

---
