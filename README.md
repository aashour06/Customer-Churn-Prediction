# 📊 Customer Churn Prediction using Artificial Neural Networks (ANN)

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![uv](https://img.shields.io/badge/Package%20Manager-uv-DE5FE9?style=for-the-badge&logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)](#)

An end-to-end Deep Learning machine learning project that predicts telecommunication customer churn using an **Artificial Neural Network (ANN)** wrapped with **SciKeras** and fine-tuned via **GridSearchCV** in a unified **Scikit-Learn Pipeline**.

---

## 📌 Table of Contents

- [Business Problem](#-business-problem)
- [Dataset Overview](#-dataset-overview)
- [Key Insights from EDA](#-key-insights-from-eda)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Model Architecture & Hyperparameter Tuning](#-model-architecture--hyperparameter-tuning)
- [Evaluation & Results](#-evaluation--results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation with uv (Recommended)](#installation-with-uv-recommended)
  - [Alternative Installation with pip](#alternative-installation-with-pip)
  - [Running the Notebook](#running-the-notebook)
- [Roadmap & Future Enhancements](#-roadmap--future-enhancements)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 💼 Business Problem

Customer churn represents the percentage of customers who stop using a company's product or service within a given timeframe. In the telecommunications sector, acquiring new customers costs **5 to 25 times more** than retaining existing ones.

This project delivers a predictive intelligence model to:
- Identify high-risk churn customers before they cancel their subscriptions.
- Uncover underlying operational drivers behind customer attrition (e.g., contract types, charges, service dependencies).
- Enable proactive retention strategies such as targeted discounts, loyalty programs, and customized long-term contract incentives.

---

## 📁 Dataset Overview

The project utilizes the [Telco Customer Churn](data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv) dataset (IBM Sample Dataset).

- **Total Records:** 7,043 customers (7,032 clean rows after handling nulls)
- **Features:** 20 predictor columns (demographics, subscribed services, account details)
- **Target Variable:** `Churn` (`Yes` / `No`)

### Key Variables:
| Category | Features |
| :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| **Target** | `Churn` (1 = Churned, 0 = Retained) |

---

## 🔍 Key Insights from EDA

1. **Contract Type is Decisive:**
   - Customers on **Month-to-Month** contracts exhibit drastically higher churn rates compared to those on **One-Year** or **Two-Year** contracts.
2. **Tenure Matters:**
   - Churn heavily concentrates within the first **1–12 months**. Customers who stay past the first year show a significantly higher retention rate.
3. **Monthly Charges Impact:**
   - Customers paying higher monthly charges ($70–$100+) churn at substantially higher frequencies, particularly those without value-add add-on services (such as Tech Support and Online Security).
4. **Data Hygiene:**
   - `TotalCharges` contained 11 blank space strings for new customers (`tenure == 0`), which were coerced to numeric nulls and dropped. Redundant categorical states (`"No phone service"` and `"No internet service"`) were normalized into `"No"`.

---

## ⚙️ System Architecture & Pipeline

A strict, leak-free Scikit-Learn `Pipeline` combined with `ColumnTransformer` guarantees that all transformations and scaling are fit solely on training data.

```
                    Raw Data (CSV)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    Numerical Features         Categorical Features
 (tenure, MonthlyCharges,   (Contract, InternetService,
       TotalCharges)              PaymentMethod, ...)
            │                           │
            ▼                           ▼
     StandardScaler              OneHotEncoder
            │                           │
            └─────────────┬─────────────┘
                          ▼
                  ColumnTransformer
                          │
                          ▼
            Deep ANN (SciKeras Classifier)
        [Dense(16) -> Dense(16) -> Dense(1)]
                          │
                          ▼
                    Churn Prediction
```

---

## 🧠 Model Architecture & Hyperparameter Tuning

The predictive engine is an **Artificial Neural Network (ANN)** built with **TensorFlow / Keras** and interfaced through **SciKeras (`KerasClassifier`)**:

### Neural Network Design
- **Input Layer:** Matches dynamic one-hot encoded dimension (`Input(shape=(n_features,))`)
- **Hidden Layer 1:** 16 Units | ReLU Activation
- **Hidden Layer 2:** 16 Units | ReLU Activation
- **Output Layer:** 1 Unit | Sigmoid Activation
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Metric:** Accuracy

### Hyperparameter Search with `GridSearchCV`
Exhaustive cross-validation was run across 36 parameter combinations (108 fits with 3-fold CV):

```python
param_grid = {
    "ann__model__neurons_1": [16, 32, 64],
    "ann__model__neurons_2": [8, 16, 32],
    "ann__epochs": [10, 20],
    "ann__batch_size": [32, 64]
}
```

### Optimal Configuration
- **Neurons (Layer 1):** `16`
- **Neurons (Layer 2):** `16`
- **Batch Size:** `64`
- **Epochs:** `20`
- **Best CV Score:** **80.43%**

---

## 📈 Evaluation & Results

The final model was evaluated on an unseen **stratified test set (20%, 1,407 samples)**:

| Metric | Train Set (80%) | Test Set (20%) |
| :--- | :---: | :---: |
| **Accuracy** | **81.26%** | **79.67% (~80%)** |

### Test Set Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Retained (`No`)** | **0.84** | **0.90** | **0.87** | 1,033 |
| **Churned (`Yes`)** | **0.65** | **0.51** | **0.57** | 374 |
| **Macro Average** | 0.74 | 0.71 | 0.72 | 1,407 |
| **Weighted Average** | 0.79 | 0.80 | 0.79 | 1,407 |

### Business Interpretation
- The model correctly flags churners with **65% precision**, enabling retention teams to allocate outreach resources with confidence.
- To maximize customer capture, the decision threshold can be lowered from the default `0.50` to prioritize **Recall** based on business cost-benefit trade-offs.

---

## 📂 Project Structure

```plaintext
customer_churn_prediction/
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw Telco dataset
├── notebooks/
│   └── ann.ipynb                                 # Complete workflow: EDA, Pipeline, GridSearch, Evaluation
├── src/
│   └── customer_churn_prediction/
│       └── __init__.py                           # Source package module
├── .python-version                               # Python version pin (3.13)
├── pyproject.toml                                # Project metadata & dependencies
├── uv.lock                                       # Deterministic lockfile
└── README.md                                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python **3.13** or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### Installation with `uv` (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Sync the environment and install all dependencies:**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```powershell
     .venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```

### Alternative Installation with `pip`

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Running the Notebook

Launch Jupyter or your preferred IDE to inspect and run the analysis:

```bash
uv run jupyter lab
# or
uv run jupyter notebook notebooks/ann.ipynb
```

---



## 👤 Author & Acknowledgments

**Ahmed Ashour**  
- **Email:** [a7med3shour10@gmail.com](mailto:a7med3shour10@gmail.com)  
- **Dataset:** Provided by IBM Cognitive Class via Kaggle.

---
*If you found this project helpful, feel free to give it a ⭐ on GitHub!*
