# Car Price Prediction Using Machine Learning

## Project Overview

This project is a machine learning application that predicts the resale price of used cars based on features such as brand, model, year, kilometers driven, fuel type, transmission, and ownership history.

The system is built using regression algorithms trained on a real-world used car dataset.

---

## Objective

To build a regression model that accurately estimates the price of a used car based on input features.

---

## Dataset Description

The dataset contains the following features:

- Brand
- Model
- Year
- Kilometers Driven
- Transmission
- Owner
- Fuel Type
- Price (Target Variable)

---

## Data Preprocessing

The following preprocessing steps were applied:

- Removal of duplicate records
- Handling missing values
- Cleaning numeric columns
- Converting categorical variables into numerical format
- One-hot encoding for categorical features
- Log transformation of target variable (Price)
- Outlier handling using quantile clipping
- Feature scaling using StandardScaler

---

## Exploratory Data Analysis (EDA)

The following analyses were performed:

- Distribution plots for numerical features
- Boxplots for outlier detection
- Count plots for categorical variables
- Scatter plots for feature relationships
- Correlation heatmap
- Bar plots for average price comparison

---

## Models Used

The following regression models were trained and evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

---

## Model Performance

| Model              | MAE   | RMSE  | R2 Score |
|-------------------|-------|-------|----------|
| Linear Regression | 0.198 | 0.291 | 0.894    |
| Ridge Regression  | 0.198 | 0.291 | 0.894    |
| Lasso Regression  | 0.221 | 0.311 | 0.879    |
| Random Forest     | 0.302 | 0.408 | 0.791    |

Best Performing Model: Ridge Regression

---

## Model Saving

```python
import joblib

joblib.dump({
    "model": model,
    "columns": X.columns,
    "scaler": scaler
}, "final_model.pkl")
How to Run the Project
Clone the Repository
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
Install Dependencies
pip install -r requirements.txt
Run the Application
streamlit run app.py
Requirements
Python 3.8+
pandas
numpy
scikit-learn
streamlit
joblib
Project Structure
car-price-prediction/
│
├── app.py
├── final_model.pkl
├── requirements.txt
├── dataset.csv
└── README.md
Author

Arun Sundar
