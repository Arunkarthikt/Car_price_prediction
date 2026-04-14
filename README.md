Car Price Prediction Using Machine Learning
Project Overview

This project is a machine learning application that predicts the resale price of used cars based on features such as brand, model, year, kilometers driven, fuel type, transmission, and ownership history.

The system is built using regression algorithms and trained on a real-world dataset of used cars.

Objective

To build a regression model that can accurately estimate the price of a used car based on input features.

Dataset Description

The dataset contains the following features:

Brand
Model
Year
Kilometers Driven
Transmission
Owner
Fuel Type
Price (Target Variable)
Data Preprocessing

The following preprocessing steps were applied:

Removal of duplicate records
Handling missing values
Cleaning numeric columns
Converting categorical variables into numerical format
One-hot encoding for categorical features
Log transformation of the target variable (Price)
Outlier handling using quantile clipping
Feature scaling using StandardScaler
Exploratory Data Analysis

The following analyses were performed:

Distribution plots for numerical features
Boxplots for outlier detection
Count plots for categorical variables
Scatter plots for feature relationships
Correlation heatmap
Bar plots for average price comparisons
Models Used

The following regression models were trained and evaluated:

Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
Model Performance
Model	MAE	RMSE	R2 Score
Linear	0.198	0.291	0.894
Ridge	0.198	0.291	0.894
Lasso	0.221	0.311	0.879
Random Forest	0.302	0.408	0.791

Best performing model: Ridge Regression

Model Saving

The trained model and preprocessing objects are saved using joblib:

joblib.dump({
    "model": model,
    "columns": X.columns,
    "scaler": scaler
}, "final_model.pkl")
How to Run the Project
Install dependencies
pip install -r requirements.txt
Run the application
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
└── dataset.csv
Future Improvements
Improve accuracy using advanced models like XGBoost or LightGBM
Add better feature engineering
Deploy as a full web application with backend API
Add user authentication system
Author

Arun Sundar
