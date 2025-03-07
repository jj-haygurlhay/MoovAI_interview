import pandas as pd
import numpy as np
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils import preprocess_Data

#load the dataset
df = pd.read_csv("stores_sales_forecasting.csv", encoding="utf-8" , encoding_errors="ignore")

df = preprocess_Data(df)
#SHAP

X = df.drop(columns=['Profit'])  # Drop target column
X = X.astype({col: int for col in X.select_dtypes('bool').columns})
y = df['Profit']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train a RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

#Initialize SHAP TreeExplainer (for random forest Regressor)
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_test)

#Plot SHAP summary plot
shap.summary_plot(shap_values, X_test)