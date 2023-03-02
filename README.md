# Loan Prediction

This project is aimed at automating the loan eligibility process based on customer details provided while filling an online application form. 
The details include gender, marital status, education, number of dependents, income, loan amount, and credit history. The goal is to identify customer segments that are eligible for a loan amount so that they can be specifically targeted.

# Problem Statement

The company wants to automate the loan eligibility process (real-time) based on customer details provided while filling the online application form. These details are:

1. Gender
2. Marital Status
3. Education
4. Number of Dependents
5. Income
6. Loan Amount
7. Credit History

To automate this process, they have given a problem to identify the customer segments that are eligible for a loan amount so that they can be specifically targeted.

# Solution

To solve this problem, we will build a machine learning model to identify eligible customers for a loan amount. We will use the following machine learning algorithms:

1. Logistic Regression
2. K-Nearest Neighbour (KNN)
3. Decision Tree
4. Random Forest
5. XGBClassifier

# Libraries Used

We have used the following libraries in this project:

1. Pandas
2. NumPy
3. Matplotlib
4. Seaborn
5. Scikit-learn
6. XGBoost

# Data Gathering and Reading

The loan dataset is downloaded from Kaggle using the opendatasets library. 
The data is stored in a CSV file.
We use the Pandas library to read the CSV file into a dataframe named loan_dataset.

# Exploratory Data Analysis (EDA)

We perform EDA to understand the data and identify any patterns, trends, or relationships that may exist in the data. 
We visualize the data using different plots, such as count plots, pie charts, histograms, box plots, and violin plots.

# Categorical Columns Visualization

We visualize the categorical columns using count plots and pie charts. The categorical columns in the loan dataset are:

1. Gender
2. Married
3. Dependents
4. Education
5. Self_Employed
6. Property_Area
7. Loan_Status

# Numerical Columns Visualization

We visualize the numerical columns using box plots, violin plots, and histograms. The numerical columns in the loan dataset are:

1. ApplicantIncome
2. CoapplicantIncome
3. LoanAmount
4. Loan_Amount_Term
5. Credit_History

# Data Preprocessing

We preprocess the data to handle missing values, encode categorical variables, and scale numerical features. 
We also use the SMOTE technique to handle class imbalance in the target variable.

# Model Building

We build machine learning models using the following algorithms:

1. Logistic Regression
2. K-Nearest Neighbour (KNN)
3. Decision Tree
4. Random Forest
5. XGBClassifier

We use the **scikit-learn library to train and evaluate the models.** 
We also use cross-validation to estimate the models' performance and tune their hyperparameters.

# Model Evaluation

We evaluate the models using **accuracy**, **confusion matrix**, **classification report**.
We select the best-performing model based on these metrics.

# Conclusion

In conclusion, we build a machine learning model to automate the loan eligibility process based on customer details. 
We preprocess the data, build five different models, and evaluate them using various metrics. 
Finally, we select the best-performing model based on the evaluation metrics. 

 
