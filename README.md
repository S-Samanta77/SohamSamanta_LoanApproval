# Loan Approval Prediction & Risk Dashboard

## Overview

This project is an academic Data Analytics and Machine Learning application that analyses a synthetic loan application dataset, builds a binary classification model to predict loan approval outcomes, and delivers an interactive Streamlit dashboard for exploration and individual prediction.

---

## Problem Statement

Financial institutions process thousands of loan applications and must evaluate applicant risk consistently. This project explores how applicant demographics, financial history, and loan characteristics relate to approval decisions, and demonstrates how a machine learning classifier can predict outcomes based on these factors.

> **Important:** The dataset used is **synthetic** and does not represent real applicant data. All predictions and results are for educational and analytical demonstration only. This system must **not** be used for real financial decisions.

---

## Objectives

- Analyse loan application data to identify patterns and trends
- Identify features most associated with loan approval or rejection
- Build a binary classification model (Random Forest) for loan approval prediction
- Evaluate model performance using standard classification metrics
- Provide an interactive analytics dashboard with KPIs and visualisations
- Allow users to test individual loan applications through a prediction interface

---

## Dataset

| Attribute       | Detail |
|----------------|--------|
| **Name**       | Loan Approval Classification Dataset |
| **Source**     | [Kaggle – taweilo/loan-approval-classification-data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data) |
| **Records**    | ~45,000 |
| **Variables**  | 14 |
| **Target**     | `loan_status` (1 = Approved, 0 = Rejected) |
| **Nature**     | **Synthetic** – generated for classification benchmarking |

### Features

| Column | Description |
|--------|-------------|
| `person_age` | Applicant age |
| `person_gender` | Applicant gender |
| `person_education` | Highest education level |
| `person_income` | Annual income ($) |
| `person_emp_exp` | Employment experience (years) |
| `person_home_ownership` | Home ownership status |
| `loan_amnt` | Requested loan amount ($) |
| `loan_intent` | Purpose of the loan |
| `loan_int_rate` | Interest rate (%) |
| `loan_percent_income` | Loan amount as a fraction of income |
| `cb_person_cred_hist_length` | Credit history length (years) |
| `credit_score` | Applicant credit score |
| `previous_loan_defaults_on_file` | Whether prior defaults exist (Yes/No) |
| `loan_status` | **Target**: 1 = Approved, 0 = Rejected |

---

## Dataset Setup

1. Download the dataset from Kaggle:  
   [https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

2. If the downloaded file is not already named `loan_data.csv`, rename it to:
   ```
   loan_data.csv
   ```

3. Place `loan_data.csv` in the **same folder** as `SohamSamanta_LoanApproval.py`.

---

## Technologies

- **Python 3.9+**
- **Pandas** – data loading, cleaning, analysis
- **NumPy** – numerical operations
- **Matplotlib** – visualisation
- **Seaborn** – statistical visualisation
- **Scikit-learn** – preprocessing, model training, evaluation
- **Streamlit** – interactive web dashboard

---

## Machine Learning Methodology

- **Task:** Binary classification (Approved / Rejected)
- **Model:** Random Forest Classifier
- **Preprocessing:**
  - Numerical features: Standard Scaling
  - Categorical features: One-Hot Encoding
  - Both steps unified in a single `sklearn.pipeline.Pipeline`
- **Split:** 80% train / 20% test, stratified, fixed `random_state=42`
- **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Prediction output:** Class label + model-predicted approval probability
- **Caching:** The trained pipeline is cached with `@st.cache_resource` to prevent retraining on every Streamlit rerun

---

## Dashboard Features

### Analytics Dashboard
- KPI cards: total applications, approved/rejected counts, approval rate, average loan, income, credit score
- Loan approval distribution chart
- Income and credit score distributions by approval status
- Approval rate by education and home ownership
- Approval rate by loan intent
- Credit score vs loan amount scatter plot
- Loan-to-income distribution
- Correlation heatmap
- Feature importance chart
- Automatically computed dataset insights

### Model Evaluation
- Accuracy, Precision, Recall, F1-score metrics
- Confusion matrix visualisation
- Full classification report

### Loan Prediction
- Form-based user input (age, income, credit score, etc.)
- Instant prediction: Approved or Rejected
- Model-predicted approval probability (%)
- Top features considered by the model

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run SohamSamanta_LoanApproval.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

---

## Project Structure

```
PROJECT_FOLDER/
├── SohamSamanta_LoanApproval.py       # Complete application (single Python file)
├── loan_data.csv         # Dataset (download from Kaggle)
├── requirements.txt      # Python dependencies
├── readme.md             # This file
└── SohamSamanta_ProjectReport.docx   # Formal project report
```

---

## Model Evaluation Summary

Performance is evaluated on a held-out 20% test set. Expected approximate metrics on this synthetic dataset:

| Metric    | Value     |
|-----------|-----------|
| Accuracy  | ~92–94%   |
| Precision | ~90–93%   |
| Recall    | ~90–93%   |
| F1 Score  | ~91–93%   |

*Exact values depend on the dataset version downloaded from Kaggle.*

---

## Limitations

- The dataset is **synthetic** and may not reflect the complexity of real-world lending data
- The model is intended for **educational purposes only**
- High accuracy on synthetic data does not guarantee performance on real applications
- Feature importance reflects predictive utility in the trained model, not causal relationships
- No hyperparameter optimisation has been applied beyond basic configuration

---

## Synthetic Data Disclaimer

> This project uses the **Loan Approval Classification Dataset** from Kaggle, which is a **synthetically generated dataset**. It does not contain real personal financial data. All analytical findings, model predictions, and insights are for educational and demonstrative purposes only. This project must **not** be used to make actual lending or financial decisions.

---

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
- Scikit-learn documentation: https://scikit-learn.org
- Streamlit documentation: https://docs.streamlit.io
