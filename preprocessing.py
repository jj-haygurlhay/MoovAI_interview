import pandas as pd
import numpy as np
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from utils import kruskal_test, cramers_v_matrix, plot_cramers_v_heatmap

#load the dataset
df = pd.read_csv("stores_sales_forecasting.csv", encoding="utf-8" , encoding_errors="ignore")

#we are going to analyze features and perform some feature selection and feature engineering 

#drop the Row ID and other non pertinant features (we drop country and category since theres only one value for each)
df = df.drop(columns=["Row ID", 'Order ID', 'Customer Name', 'State', 'Postal Code', 'Ship Date', 'Country', 'Category'], axis = 1)

#first I handle the inconsistant data type of the order date and engineer it into day, year, week, day of week

#convert Order Data to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

#extract year, month and day
df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.month
df['Order Day'] = df['Order Date'].dt.day
df['Order Day of Week'] = df['Order Date'].dt.dayofweek

#so the date has been transformed into features more digestable
#drop order date since it has been transformed
df = df.drop(['Order Date'], axis=1)

#now we check the importance of customer ID

#check importance of customer ID check how many unique customers there are
#print(f"number of unique customers: {df['Customer ID'].nunique()}")
#check if there are many repeat customers
#repeat_customers = df.groupby('Customer ID').size().reset_index(name='Purchase Count')
#repeat_customers = repeat_customers[repeat_customers['Purchase Count'] > 1]
#print(f"number of repeat customers {repeat_customers.shape[0]}")  # Number of repeat customers#print head of dataframe

#547 out of 707 customers are repeat customers
#could be important feature in predicting profit 
#so i changed customer ID to purchase count (simpler easier to handle)
purchase_counts = df.groupby('Customer ID').size().reset_index(name='Purchase Count')
df = df.merge(purchase_counts, on='Customer ID', how='left')
# Drop the original 'Customer ID' column
df = df.drop('Customer ID', axis=1)

# now I handle the Product Name
# there were like 380 product names which is too many and could lead to overfitting so it is dropped
df = df.drop('Product Name', axis=1)

#feature engineering (adding Discounted_Sales, Price_per_Unit, Discount_Impact, Log_Sales and Profit_Margin)
df["Discounted_Sales"] = df["Sales"] * (1 - df["Discount"])
df["Price_per_Unit"] = df["Sales"] / df["Quantity"]
df["Discount_Impact"] = df["Sales"] - df["Discounted_Sales"]
df["Log_Sales"] = np.log1p(df["Sales"])  # log1p to handle zeros safely
df["Profit_Margin"] = (df["Profit"] / df["Sales"]) * 100

#now we must check which features are correlated with the profit and which are correlated with each other
#calculate Pearson correlation for numerical features
numerical_features = ['Sales', 'Quantity', 'Discount', 'Order Year', 'Order Month', 'Order Day', 'Order Day of Week', 
                      'Purchase Count', 'Discounted_Sales', 'Price_per_Unit', 'Discount_Impact', 'Log_Sales', 'Profit_Margin']
correlation_matrix = df[numerical_features + ['Profit']].corr()

#display correlation with Profit
#print(correlation_matrix['Profit'].sort_values(ascending=False))
#Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap for Numerical Features')
plt.show()

#this plot showed that only sales and discount and quantity (through the sales) are correlated with the profit
df = df.drop(columns=['Order Year', 'Order Month', 'Order Day', 'Order Day of Week', 'Purchase Count'], axis = 1)

#Drop the same columns from numerical_features and regraph the correlation matrix for clarity
numerical_features = [col for col in numerical_features if col not in ['Order Year', 'Order Month', 'Order Day', 
                                                                       'Order Day of Week', 'Purchase Count']]
correlation_matrix = df[numerical_features + ['Profit']].corr()
#display correlation with Profit

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap for Numerical Features II')
plt.show()

# print(df.head())
columns_to_drop = ['Log_Sales','Price_per_Unit', 'Sales']
df = df.drop(columns=columns_to_drop, axis = 1)
#Drop the same columns from numerical_features and regraph the correlation matrix for clarity
numerical_features = [col for col in numerical_features if col not in columns_to_drop]
correlation_matrix = df[numerical_features + ['Profit']].corr()
#display correlation with Profit

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap for Numerical Features III')
plt.show()
#now that we performed a little feature engineering and selection it is time to handle the categorical features
categorical_features = ['Ship Mode', 'Segment', 'City', 'Region', 'Product ID', 'Sub-Category']

#count unique values in each categorical feature
unique_counts = df[categorical_features].nunique()

#display the results
# print(f"{unique_counts}")
#this showed there was only one country and one category so we dropped them
#kruskal wallis test since not all categorical features are normally distributed and some have a high cardinality
results = {}
for feature in categorical_features:
    p_value = kruskal_test(feature, df)
    results[feature] = p_value

# Display results
for feature, p in results.items():
    print(f'{feature}: p = {p:.75f}')

#results: 
# Ship Mode: p = 0.640858963297465056285773243871517479419708251953125000000000000000000000000
# Segment: p = 0.742699787038433312247320827736984938383102416992187500000000000000000000000
# City: p = 0.000000000000000000000000000000000000000000000000000023089338702713992344369
# Region: p = 0.000000000000000000000000000044424002269190497763936303631559434345376292876
# Product ID: p = 0.000000000000000000000000000022376522647010394838752697270770256555481716113
# Sub-Category: p = 0.000000000000000000000093684050427002657401091624321469457277740266003712670
# -> drop ship mode and segment
df = df.drop(columns=['Ship Mode', 'Segment'], axis=1)

categorical_features = ['City', 'Region', 'Product ID', 'Sub-Category']

#using cramers V to determine correlation between these remaining categorical features 
cramers_v_results = cramers_v_matrix(df, categorical_features)
plot_cramers_v_heatmap(cramers_v_results)

#results showed city and region and product ID and sub category are correlated
df = df.drop(columns=['City','Product ID'], axis=1)
#now we are using mutual info regression to test feature importance 
print(df.head(10))

#check data types
print(df.info())

#SHAP
#convert Region and Sub-Category into numerical values using one-hot encoding
categorical_features = ['Region', 'Sub-Category']

#one-Hot Encode categorical variables
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

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


