# 🧠 Mental Health Risk Prediction System

This project aims to predict the risk of mental health issues in individuals using machine learning techniques. The solution is built with a **LightGBM** model, achieving a Kaggle competition score of **92.17%**, and incorporates a user-friendly **Streamlit web app** for practical use.

---

## 📜 Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Modeling Pipeline](#modeling-pipeline)
5. [Performance](#performance)
6. [Streamlit Web App](#streamlit-web-app)


---

## Overview

Mental health is a pressing global concern, and early detection plays a critical role in providing timely support. This system leverages a machine learning model trained on a synthetic dataset derived from the Depression Survey/Dataset for Analysis. The solution:
- Predicts if an individual is **"At Risk"** or **"Not at Risk"** of mental health issues.
- Features robust preprocessing, model training, and hyperparameter optimization.
- Offers a streamlined **Streamlit web app** for easy interaction.

---

## Dataset

### Source
The dataset is a synthetic variant of the **Depression Survey/Dataset for Analysis**, generated using deep learning. The train and test datasets share close distributions with the original data.

### Structure
The dataset contains **20 features** split into demographic, academic, and lifestyle factors. Some highlights:
- **Rows**: 140,700 (train set)
- **Features**: 20 (8 numerical, 10 categorical, and 2 target-related)
- **Target Variable**: `Depression` (binary: 0 for "Not at Risk", 1 for "At Risk")

### Example Columns
- `Gender`: Male/Female/Other.
- `Sleep Duration`: Less than 5 hours / 5-6 hours / More than 8 hours.
- `Financial Stress`: Numeric scale representing financial difficulty.
- `Depression`: Target variable for prediction.

---

## Features

The dataset includes features grouped into the following categories:

1. **Demographic**: `Age`, `Gender`, `City`, `Profession`.
2. **Academic**: `Degree`, `CGPA`, `Work/Study Hours`, `Academic Pressure`, `Study Satisfaction`.
3. **Lifestyle**: `Dietary Habits`, `Sleep Duration`, `Job Satisfaction`, `Financial Stress`, `Family History of Mental Illness`.
4. **Target**: `Depression` (binary classification).

---

## Modeling Pipeline

The pipeline includes:

1. **Data Preprocessing**:
   - Handling missing values: Numerical features imputed with medians, categorical features with modes.
   - Feature engineering: Added binary indicators for missing data and categorized `Age` into groups.
   - Encoding categorical variables with **OneHotEncoder**.
   - Scaling numerical variables using **StandardScaler**.

2. **Model Training**:
   - Base model: **LightGBMClassifier**.
   - Hyperparameter tuning via **GridSearchCV**.

3. **Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-Score

---

## Performance

### Model Accuracy:
- **Training Accuracy**: 93.11%
- **Test Accuracy**: 91.77%
- **Kaggle Submission Score**: **92.17%**

### Evaluation Metrics:
| Metric       | Score   |
|--------------|---------|
| Precision    | 90%     |
| Recall       | 89%     |
| F1-Score     | 90%     |

---

## Streamlit Web App

The interactive **Streamlit web app** allows users to input their data and receive predictions instantly.  

### Features:
- User-friendly interface for data input.
- Displays predictions in real-time.
- Easy to deploy locally or on cloud platforms like Heroku or AWS.
- Libraries: LightGBM, pandas, scikit-learn, Streamlit


