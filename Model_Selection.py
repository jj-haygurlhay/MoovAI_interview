import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from utils import preprocess_Data
#load dataset
df = pd.read_csv("stores_sales_forecasting.csv", encoding="utf-8" , encoding_errors="ignore")
#preprocess data
df = preprocess_Data(df=df)
#split data into X and y (features and predictions)
X = df.drop(columns=['Profit'])
y = df['Profit']
#split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RootMeanSqErr = []
R2scores = []

#as a baseline we will test Linear Regression
# Train the model
model_LR = LinearRegression()
model_LR.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_LR.predict(X_test)

#mean squared error
mse = np.sqrt(mean_squared_error(y_test, y_pred))
# range_y = np.max(y_test)-np.min(y_test)
# mse = mse/range_y

print("Linear Regression RMSE:", mse)
r2 = r2_score(y_test, y_pred)
print("Linear Regression R²:", r2)
#add to lists for comparison
RootMeanSqErr.append(('Linear Regression (baseline)', mse))
R2scores.append(('Linear Regression (baseline)', r2))

#RandomForestRegressor
model_RF = RandomForestRegressor(n_estimators=100, random_state=42)
model_RF.fit(X_train, y_train) #train model

# Predict and evaluate
y_pred = model_RF.predict(X_test)
mse_RF = np.sqrt(mean_squared_error(y_test, y_pred))
# range_y = np.max(y_test)-np.min(y_test)
# mse_RF = mse_RF/range_y
print("Random Forest RMSE:", mse_RF)
r2_rf = r2_score(y_test, y_pred)
print("Random Forest R²:", r2_score(y_test, y_pred))
#add results to lists
RootMeanSqErr.append(('Random Forest', mse_RF))
R2scores.append(('Random Forest', r2_rf))

#XgBoost
model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_xgb.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_xgb.predict(X_test)
mse_xgb = np.sqrt(mean_squared_error(y_test, y_pred))
# range_y = np.max(y_test)-np.min(y_test)
# mse_xgb = mse_xgb/range_y
print("XGB RMSE:", mse_xgb)
r2_xgb = r2_score(y_test, y_pred)
print("XGB R²:", r2_xgb)

#add results to lists
RootMeanSqErr.append(('XGBoost', mse_xgb))
R2scores.append(('XGBoost', r2_xgb))

#Support Vector Regression
# Scale the data (SVM is sensitive to feature scales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # Train the model
model_SVR = SVR(kernel='rbf')  # You can try 'linear' or 'poly' kernels too
model_SVR.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model_SVR.predict(X_test_scaled)
mse_SVR = np.sqrt(mean_squared_error(y_test, y_pred))
# range_y = np.max(y_test)-np.min(y_test)
# mse_SVR = mse_SVR/range_y
print("SVR RMSE:", mse_SVR)
r2_SVR = r2_score(y_test, y_pred)
print("SVR R²:", r2_SVR)

#add results to lists
RootMeanSqErr.append(('SVR', mse_SVR))
R2scores.append(('SVR', r2_SVR))

#KNN Regressor
model_KNN = KNeighborsRegressor(n_neighbors=5)  # Adjust n_neighbors as needed
model_KNN.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_KNN.predict(X_test)
mse_KNN = np.sqrt(mean_squared_error(y_test, y_pred))
# range_y = np.max(y_test)-np.min(y_test)
# mse_KNN = mse_KNN/range_y
print("KNN RMSE:", mse_KNN)
r2_KNN = r2_score(y_test, y_pred)
print("KNN R²:", r2_KNN)

#add results to lists
RootMeanSqErr.append(('KNN', mse_KNN))
R2scores.append(('KNN', r2_KNN))

#Light GBM
# Train the model
model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_lgb.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_lgb.predict(X_test)
mse_lgb = np.sqrt(mean_squared_error(y_test, y_pred))
# range_y = np.max(y_test)-np.min(y_test)
# mse_lgb = mse_lgb/range_y
print("RMSE:", mse_lgb)
r2_lgb = r2_score(y_test, y_pred)
print("Light GBM R²:", r2_lgb)

#add results to lists
RootMeanSqErr.append(('Light GBM', mse_lgb))
R2scores.append(('Light GBM', r2_lgb))

#graph results for analysis

#start with mean squared error
#unpack tuples into two lists of categories and values
model_names, errors = zip(*RootMeanSqErr)
#create and plot the bar graph
plt.bar(model_names, errors, color = 'skyblue', width=0.6)
# Add labels and title
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error for each Model')
# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Show the plot
plt.show()

#now we plot the R2 scores
#unpack tuples into two lists of categories and values
model_names, r2s = zip(*R2scores)
#create and plot the bar graph
plt.bar(model_names, r2s, color = 'skyblue', width=0.6)
# Add labels and title
plt.xlabel('Models')
plt.ylabel('R² scores')
plt.title('R² score for each Model')
# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Show the plot
plt.show()
