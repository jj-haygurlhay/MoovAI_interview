import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from utils import preprocess_Data
import optuna
import json

#load dataset
df = pd.read_csv("stores_sales_forecasting.csv", encoding="utf-8" , encoding_errors="ignore")
#preprocess data
df = preprocess_Data(df=df)
#split data into X and y (features and predictions)
X = df.drop(columns=['Profit'])
y = df['Profit']
#split data for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#define objective function for optuna to use (identify hyperparameters to be tuned)
def objective(trial):
    # Suggest values for hyperparameters
    param = {
        'objective': 'reg:squarederror',  # For regression
        'eval_metric': 'rmse',  # RMSE as evaluation metric
        'max_depth': trial.suggest_int('max_depth', 3, 12),  # max depth of trees
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),  # log scale for learning rate
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # number of boosting rounds
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),  # fraction of data used for each tree
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),  # fraction of features
        'gamma': trial.suggest_loguniform('gamma', 1e-3, 1e-1),  # regularization
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1e-1),  # L1 regularization
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1e-1)  # L2 regularization
    }

    # Train XGBoost model
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)

    # Predict on validation set
    preds = model.predict(X_valid)
    
    # Calculate the RMSE
    #for nomalizing the RMSE we would divide by (np.max(y_valid)-np.min(y_valid)) 
    # but for interpretability we will stick to the RSME
    rmse = np.sqrt(mean_squared_error(y_valid, preds))

    return rmse

# Create and run the study
study = optuna.create_study(direction='minimize')  # Minimize RMSE
study.optimize(objective, n_trials=50)  # Number of trials to run

trials = study.trials_dataframe()

# Sort by value (RMSE) and select top 5
top_trials = trials.nsmallest(5, 'value')
print(top_trials[['value', 'params_max_depth', 'params_learning_rate', 
                  'params_n_estimators', 'params_subsample', 'params_colsample_bytree']])

# Extract RMSE values and trial numbers
trial_numbers = top_trials.index
rmse_values = top_trials['value']

plt.figure(figsize=(8, 5))
plt.bar(trial_numbers, rmse_values, color='skyblue')
plt.xlabel("Trial Number")
plt.ylabel("RMSE")
plt.title("Top 5 Best Hyperparameter Sets (Lowest RMSE)")
plt.xticks(trial_numbers)
plt.show()

# Get the best hyperparameters
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

with open("best_hyperparameters_XGBoost.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("Best hyperparameters saved to best_hyperparameters_XGBoost.json")
