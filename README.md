# Breast Cancer Cell Data Analysis and Classification Project

## Overview
This project analyzes and classifies breast cancer cell data using machine learning techniques.  
The dataset consists of 569 samples with 32 features, which are explored, preprocessed, and used to train and evaluate multiple classification models.

## Dataset
- **Source**: [Breast Cancer Data Set - Kaggle](https://www.kaggle.com/datasets/erdemtaha/cancer-data)
- **Target column**: `diagnosis` (Malignant = M, Benign = B)
- **Features**: 32 attributes describing cell characteristics
- **Samples**: 569
- **Missing values**: None

## 1. Data Preparation
- Import required libraries (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn).
- Load dataset from CSV file.
- Inspect dataset structure using `df.info()`.
- Generate statistical summary using `df.describe()`.

## 2. Exploratory Data Analysis (EDA)
- **Class distribution**: Checked counts of Malignant and Benign samples.
- **Scatter plot**: Visualized `radius_mean` vs `texture_mean`.
- **Box plot**: Compared `area_mean` across cancer types.
- **Histogram**: Displayed distribution of `perimeter_mean`.
- **Heatmap**: Examined correlations between features to identify redundancy.

## 3. Data Splitting and Feature Scaling
- Split dataset into:
  - **X**: Features (independent variables).
  - **y**: Target (diagnosis, encoded as 1 for Malignant, 0 for Benign).
- Applied `train_test_split()` with stratification:
  - 70% training set, 30% testing set.
- Applied `StandardScaler` for normalization.

## 4. Model Training and Evaluation
Defined helper functions:
- `plot_confusion_matrix()`: Displays confusion matrix.
- `evaluate_model()`: Trains model, evaluates metrics, and plots ROC curve.

Evaluation metrics:
- **Accuracy**: Proportion of correct predictions.
- **Precision**: Proportion of predicted positives that are correct.
- **Recall (Sensitivity)**: Proportion of actual positives correctly identified.
- **F1 Score**: Harmonic mean of precision and recall.
- **AUC (ROC Curve)**: Ability of the model to distinguish between classes.

## 5. Model Selection
The following machine learning models were built and evaluated:
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

For imbalanced data handling, `class_weight='balanced'` was used where applicable.

## 6. Model Evaluation
- Confusion Matrix for each model.
- ROC Curve comparison.
- Reported metrics: Accuracy, Precision, Recall, F1 Score, AUC.

## 7. Model Saving and Testing
- Trained model saved using `pickle` (`model.pkl`).
- Model reloaded and tested with sample inputs from the test set.

## Results Summary
- Models achieved high performance with accuracy around 95–98%.
- ROC curves indicated good separability between Malignant and Benign cases.
- Random Forest and SVM performed slightly better than Logistic Regression.

## Reference
- Dataset: [Breast Cancer Data Set - Kaggle](https://www.kaggle.com/datasets/erdemtaha/cancer-data)
