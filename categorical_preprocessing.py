import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import scipy.stats as stats
from utils import kruskal_test, cramers_v_matrix, plot_cramers_v_heatmap

#load the dataset
df = pd.read_csv("stores_sales_forecasting.csv", encoding="utf-8" , encoding_errors="ignore")

#we are going to analyze features and perform some feature selection and feature engineering 

#drop the Row ID and other non pertinant features (we drop country and category since theres only one value for each)
df = df.drop(columns=["Row ID", 'Order ID', 'Customer Name', 'State', 'Postal Code', 'Ship Date', 'Country', 'Category'], axis = 1)
df = df.drop(['Order Date'], axis=1)
# Drop the original 'Customer ID' column
df = df.drop('Customer ID', axis=1)

# now I handle the Product Name
# there were like 380 product names which is too many and could lead to overfitting so it is dropped
df = df.drop('Product Name', axis=1)

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
