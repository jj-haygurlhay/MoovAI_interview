import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from utils import preprocess_Data
import json
import joblib

#Load the trained model
try:
    model = joblib.load("xgb_model.pkl")
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Model file 'xgb_model.pkl' not found! Ensure you've trained and saved the model.")
    exit()

#Load best hyperparameters (for reference)
try:
    with open("best_hyperparameters_XGBoost.json", "r") as f:
        best_params = json.load(f)
    print(f"🔧 Using best hyperparameters: {best_params}")
except FileNotFoundError:
    print("⚠️ Warning: 'best_hyperparameters_XGBoost.json' not found. Proceeding without it.")

#Load hypothetical test data
try:
    df_test = pd.read_csv("example_test_set.csv", encoding="utf-8", encoding_errors="ignore")
    print(f"📂 Loaded test dataset with {df_test.shape[0]} samples and {df_test.shape[1]} features.")
except FileNotFoundError:
    print("❌ Test dataset 'example_test_set.csv' not found! Provide a valid test set.")
    exit()

# Preprocess test data
df_test = preprocess_Data(df_test)

#Check if 'Profit' exists for RMSE evaluation
if "Profit" in df_test.columns:
    X_test = df_test.drop(columns=['Profit'])
    y_test = df_test["Profit"]
    print("📊 'Profit' column found. Will compute RMSE.")
else:
    X_test = df_test
    y_test = None
    print("⚠️ 'Profit' column NOT found. Predictions will be made, but RMSE won't be calculated.")

#Make predictions
predictions = model.predict(X_test)

#Display results
print("\n🔮 Predictions on new (simulated) data:")
for i, pred in enumerate(predictions[:10]):  # Show only first 10 for readability
    print(f"Sample {i+1}: Predicted Profit = {pred:.4f}")

#Save predictions to CSV
df_results = df_test.copy()
df_results["Predicted_Profit"] = predictions
df_results.to_csv("predictions.csv", index=False)
print("💾 Predictions saved to 'predictions.csv'.")

#Compute RMSE if ground truth is available
if y_test is not None:
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"\n✅ RMSE on test data: {rmse:.4f}")

