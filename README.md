# MoovAI_interview
Interview for data scientist position at MoovAI

In this Repository you will find files related to a mini project (for an interview) using ML to predict profits

- preprocessing.py:
  - here you will find my process for feature selection.  I make a correlation matrix for the numerical features to see how much each feature influences the profit in order to decide which features to eliminate, then I made two more correlation graphs as I eliminated more features to really narrow down to the most important features

  - I also perform feature engineering creating new more relevant features such as Discount_Impact, Discounted_Sales, Profit_Margin as well as other which get eliminated later on

  - You will also see some tests I did which did not lead anywhere but I left them there to show my thought process (extracting the order month, day of the week, year, etc., which I do not use in the end)

  - After the numerical features I look use the kruskal wallis test in order to test which categorical features are relevant to the profit, I choose this test because it works for non linear variables and variables which do not follow a normal distribution

  - After eliminating Ship Mode and Segment I used Cramers V test to determine correlation between the remaining categorical features
  
  - Then I used one shot encoding for the remaining categorical features to convert them into numerical values 

  - Then after eliminating the redundant categorical features I use SHAP to look at the feature importance of the remaining features 

- Model_Selection.py: 
    - in this file I tested several machine learning models and measured their Root Mean Squared Error and their R2 scores to see how effective they were at predicting the profit.  The Best performing model was XGBoost.  The other models tested were: Linear Regression, Random Forest Regressor, SVR (support vector regression), KNN (for regression) and LGB (light gradient boost) 

- hyperparameters.py: 
    - in this file I tuned the hyperparameter selection using optuna.  Optuna used Bayesian optimization and the Tree-structure Parzen Estimator algorithm.  It used an objective function that defined the hyperparameters to be tuned and hyperparameter search where it trains and evaluates the model on each set of hyperparameters.  Then it used Bayesian optimization (TPE) to guide the search for the best hyperparameters, it woudl adjust the search space to explore regions with potentially a lower RSME helping Optuna converge more quickly to an optimal set of hyperparameters.  the best hyperparameters were saved in a .json.  

- training.py: 
    - here is where I train the xgboost model using the best hyperparameters on the data (which is preprocessed using a preprocessing function I made in utils.py)
    then the model is saved afterwards as a .pkl

- inference.py:
    - this is code to be used when trying to predict the profit, or test the model on unseen data.  change the name of the csv file to make it work properly.  

- utils.py:
    - contains functions used throughout the project (such as preprocess_Data)