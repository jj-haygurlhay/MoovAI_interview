import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#load the dataset
df = pd.read_csv("stores_sales_forecasting.csv", encoding="utf-8" , encoding_errors="ignore")

#we are going to analyze features and perform some feature selection and feature engineering 

#drop the Row ID and other non pertinant features (we drop country and category since theres only one value for each)
df = df.drop(columns=["Row ID", 'Order ID', 'Customer Name', 'State', 'Postal Code', 'Ship Date', 'Country', 'Category'], axis = 1)

#first I handle the inconsistant data type of the order date and engineer it into day, year, week, day of week

#convert Order Data to datetime
# df['Order Date'] = pd.to_datetime(df['Order Date'])

# #extract year, month and day
# df['Order Year'] = df['Order Date'].dt.year
# df['Order Month'] = df['Order Date'].dt.month
# df['Order Day'] = df['Order Date'].dt.day
# df['Order Day of Week'] = df['Order Date'].dt.dayofweek

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
numerical_features_original = ['Sales', 'Quantity', 'Discount']
numerical_features = ['Sales', 'Quantity', 'Discount', 'Purchase Count', 'Discounted_Sales', 
                      'Price_per_Unit', 'Discount_Impact', 'Log_Sales', 'Profit_Margin']
correlation_matrix = df[numerical_features_original + ['Profit']].corr()

#display correlation with Profit
#print(correlation_matrix['Profit'].sort_values(ascending=False))
#Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap for Original Numerical Features')
plt.show()

#this plot showed that only sales and discount and quantity (through the sales) are correlated with the profit
df = df.drop(columns=['Purchase Count'], axis = 1)

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