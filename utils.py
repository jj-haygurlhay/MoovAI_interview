import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder

def kruskal_test(feature, data):
    groups = [data[data[feature] == group]['Profit'] for group in data[feature].unique()]
    stat, p_value = kruskal(*groups)
    return p_value

def cramers_v_matrix(df, categorical_features):
    # Computes the Cramér's V correlation matrix for categorical features in a DataFrame.
    # Parameters:
    #     df: The dataset containing categorical and numerical columns.
    #     categorical_features: List of categorical column names to analyze.
    # Returns:
    #     pd.DataFrame: A correlation matrix with Cramér’s V values.
    def cramers_v(x, y):
        # Computes Cramér's V statistic for two categorical variables
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1))) if n > 0 else 0

    # Create an empty DataFrame for the correlation matrix
    n = len(categorical_features)
    cramers_v_df = pd.DataFrame(np.zeros((n, n)), index=categorical_features, columns=categorical_features)

    # Compute pairwise Cramér’s V values
    for i in range(n):
        for j in range(i + 1, n):  # Avoid redundant calculations
            col1, col2 = categorical_features[i], categorical_features[j]
            cramers_v_df.loc[col1, col2] = cramers_v(df[col1], df[col2])
            cramers_v_df.loc[col2, col1] = cramers_v_df.loc[col1, col2]  # Symmetric matrix

    return cramers_v_df

def plot_cramers_v_heatmap(cramers_v_df):
    # Plots a heatmap of the Cramér's V correlation matrix.
    # Parameters:
    #     cramers_v_df (pd.DataFrame): DataFrame containing Cramér’s V values.
    plt.figure(figsize=(8, 6))
    sns.heatmap(cramers_v_df, annot=True, cmap="coolwarm", linewidths=0.5, vmin=0, vmax=1)
    plt.title("Cramér's V Heatmap (Categorical Feature Correlation)")
    plt.show()

def preprocess_Data(df):
    #creating new more predictive features
    df["Discounted_Sales"] = df["Sales"] * (1 - df["Discount"])
    df["Discount_Impact"] = df["Sales"] - df["Discounted_Sales"]
    df["Profit_Margin"] = (df["Profit"] / df["Sales"]) * 100
    
    #dropping all features found to be less important
    columns_to_drop = ['Row ID', 'Order ID', 'Customer Name', 'State', 'Postal Code', 
                          'Ship Date', 'Country', 'Category', 'Order Date', 'Customer ID',
                          'Product Name', 'Sales', 'Ship Mode', 'Segment', 'City','Product ID']
    df = df.drop(columns=columns_to_drop, axis = 1)
    
    #one-Hot Encode categorical variables
    categorical_features = ['Region', 'Sub-Category']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    #changing new columns into integers 
    df = df.astype({col: int for col in df.select_dtypes('bool').columns})

    return df    


