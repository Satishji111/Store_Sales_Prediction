# Store Sales Prediction

An end-to-end **Machine Learning regression project** to predict store sales using historical retail data. This project follows **industry-standard modular coding practices**, robust data preprocessing, multiple ML models with hyperparameter tuning, and a complete prediction pipeline.

---

## Project Overview

Retail businesses rely heavily on accurate sales forecasting for inventory planning, supply chain optimization, and revenue growth. This project aims to **predict `Item_Outlet_Sales`** using product-level and outlet-level attributes.

Key highlights:

* Modular & scalable project structure
* Advanced data preprocessing using pipelines
* Multiple regression models with comparison
* Hyperparameter tuning
* End-to-end prediction pipeline

---

## Problem Statement

Given historical sales data of products across different outlets, predict the **sales of a product at a particular store**.

**Target Variable:**

* `Item_Outlet_Sales`

---

## Project Structure

```bash
Store_Sales_Prediction/
│
├── artifacts/                 # Saved models, preprocessors & transformed data
│
├── notebook/
│   └── EDA.ipynb              # Exploratory Data Analysis
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading & train-test split
│   │   ├── data_transformation.py # Data cleaning, encoding & pipelines
│   │   └── model_trainer.py       # Model training & evaluation
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py      # Training pipeline
│   │   └── predict_pipeline.py    # Prediction pipeline
│   │
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   └── utils.py                   # Common utility functions
│
├── app.py                    # Application entry point
├── requirements.txt          # Project dependencies
├── setup.py                  # Package setup
└── README.md                 # Project documentation
```

---

## Technologies & Tools Used

* **Programming Language:** Python
* **Libraries:**

  * NumPy
  * Pandas
  * Scikit-learn
  * XGBoost
  * Matplotlib / Seaborn (EDA)
* **Concepts:**

  * Feature Engineering
  * Pipelines
  * Hyperparameter Tuning
  * Modular Coding
  * Model Evaluation

---

## Data Preprocessing Steps

* Handling missing values
* Mapping incorrect / inconsistent values
* Ordinal encoding (e.g. `Outlet_Size` → Small:1, Medium:2, Large:3)
* One-hot encoding for categorical features
* Numerical feature scaling
* Outlier handling using **IQR method**
* Train-test split

All transformations are handled using **Scikit-learn Pipelines** for consistency between training and prediction.

---

## Models Implemented

The following regression models are trained and evaluated:

* Linear Regression
* Polynomial Linear Regression
* Lasso Regression
* Ridge Regression
* Random Forest Regressor
* XGBoost Regressor

**Best performing model** is selected based on evaluation metrics and saved for inference.

---

## Model Evaluation Metrics

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

---

## Prediction Pipeline

The `PredictPipeline`:

* Loads the saved **preprocessor** and **trained model**
* Accepts user input features
* Applies same transformations as training
* Generates final sales prediction

A `CustomData` class is used to convert user inputs into a DataFrame format.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Satishji111/Store_Sales_Prediction.git
cd Store_Sales_Prediction
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3️. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️. Run Training Pipeline

```bash
python src/pipeline/train_pipeline.py
```

### 5️. Run Prediction Pipeline

```bash
python src/pipeline/predict_pipeline.py
```

---

## Future Improvements

* Add model explainability (SHAP / LIME)
* Build REST API using Flask/FastAPI
* Deploy on cloud (AWS / Azure / GCP)
* Integrate CI/CD pipeline
* Add unit tests

---

## Author

**Satish Yadav**
Senior Data Analyst
Data Science | Machine Learning | SQL | Python

🔗 GitHub: [https://github.com/Satishji111](https://github.com/Satishji111)


⭐ If you like this project, give it a star!

This helps others discover the project and motivates further improvements 🚀

---

## ⭐ If you like this project, give it a star!

This helps others discover the project and motivates further improvements 🚀
