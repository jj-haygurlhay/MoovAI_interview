import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from utils import preprocess_Data
import json
import joblib  # For saving the model

#load dataset
df = pd.read_csv("stores_sales_forecasting.csv", encoding="utf-8" , encoding_errors="ignore")
#preprocess data
df = preprocess_Data(df=df)
#split data into X and y (features and predictions)
X = df.drop(columns=['Profit'])
y = df['Profit']
#split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Load best hyperparameters
with open("best_hyperparameters_XGBoost.json", "r") as f:
    best_params = json.load(f)

#Train the XGBoost model
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train)

#Evaluate on validation set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Validation RMSE: {rmse}")

#Save the trained model
joblib.dump(model, "xgb_model.pkl")
print("Model saved as xgb_model.pkl")    

#Save RMSE to a log file
log_data = {"RMSE": rmse, "best_hyperparameters": best_params}
with open("training_log.json", "w") as f:
    json.dump(log_data, f, indent=4)
print("📄 Training log saved as training_log.json")

#Plot Feature Importance
feature_importance = model.feature_importances_
feature_names = X.columns

#Sort features by importance
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[sorted_idx], y=np.array(feature_names)[sorted_idx], palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.show()