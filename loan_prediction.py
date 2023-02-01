# -*- coding: utf-8 -*-
"""Loan Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vUZkM2hvyXtXtQB1lONLqiogG_6hnupQ

# **1. Problem Statement**


Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. 

These details are: 
1. Gender
2. Marital Status 
3. Education 
4. Number of Dependents 
5. Income 
6. Loan Amount 
7. Credit History 

To automate this process, they have given a problem to identify the customers segments
Those are eligible for loan amount so that they can specifically target these customers. 

**The machine learning models used:**

1. Logistic Regression
2. K-Nearest Neighbour (KNN)
3. Decision Tree
4. Random Forest
5. XGBClassifier

# 2. Importing Libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')
sns.set()

"""# 3. Data Gathering and Reading"""

#! pip install opendatasets
import opendatasets
dataset_url = 'https://www.kaggle.com/datasets/ninzaami/loan-predication'
opendatasets.download(dataset_url)

# read csv file using pandas
data_path = '/content/loan-predication/train_u6lujuX_CVtuZ9i (1).csv'
loan_dataset = pd.read_csv(data_path)
# read first five rows
loan_dataset.head()

# shape of the dataset
loan_dataset.shape
# 614 ---> rows
# 13 -----> columns

# names of the columns in the dataset
loan_dataset.columns

"""**Data types in the dataset**"""

loan_dataset.dtypes

"""**Check for categorical columns and numerical columns**"""

categorical_columns = loan_dataset.select_dtypes('object').columns.to_list()
numerical_columns = loan_dataset.select_dtypes(['int64','float64']).columns.to_list()

# categorical_columns
categorical_columns

"""**Categorical_columns**
1. Loan_ID
2. Gender
3. Married
4. Dependents
5. Education
6. Self_Employed
7. Property_Area
8. Loan_Status

"""

# numerical columns
numerical_columns

"""**Numerical_Columns**

1. ApplicantIncome
2. CoapplicantIncome
3. LoanAmount
4. Loan_Amount_Term
5. Credit_History

**Descriptive statistics**
"""

loan_dataset.describe() # for numerical columns

loan_dataset.describe(include='object') # for categorical columns

"""**Top -->Mode** 

**freq --> The number of times the mode was observed**

# **4. Expolatory Data Analysis**

**4.1 categorical_columns visualization**
"""

categorical_columns.pop(0)
categorical_columns

def categorical_plot(df,plot_kind,columns):
  plot_kind= plot_kind.lower()
  plot_function = {
      'countplot':sns.countplot,
      'pie':plt.pie,
      'violin':sns.violinplot,
      'boxplot':sns.boxplot,
      'histplot':sns.histplot
  }
  fig = plt.figure(figsize=(10,8))
  for index,column in enumerate(columns):
    axis = fig.add_subplot(3, 3, index + 1)
    if plot_kind == 'countplot':
       plot_function[plot_kind](df[column], ax=axis)
       plt.title(f'{plot_kind} for {column}')
    else:
      data = df[column].value_counts()
      label = data.index
      plot_function[plot_kind](data,labels =label,autopct ='%.f%%')
      plt.title(f'{plot_kind} plot for {column}')
  plt.tight_layout()
  plt.show()

plot_kind = ['countplot','pie']

for plot in plot_kind:
  categorical_plot(loan_dataset,plot,categorical_columns)

"""**4.2 Numerical_columns visualization**"""

def numerical_plot(df,plot_kind,columns):
  plot_kind = plot_kind.lower()
  plot_function = {
      'violin':sns.violinplot,
        'boxplot':sns.boxplot,
        'histplot':sns.histplot 
  }
  fig = plt.figure(figsize=(8,3))
  for index,column in enumerate(columns):
    axis = fig.add_subplot(1, 3, index + 1)
    if plot_kind in ['violin','boxplot']:
      plot_function[plot_kind](y=df[column], ax=axis)
      plt.title(f'{plot_kind} for {column}')
    else:
      plot_function[plot_kind](df[column], ax=axis)

  plt.tight_layout()
  plt.show

numerical_columns.pop(3)
numerical_columns.pop(4)
numerical_columns

plot_kind = ['violin','boxplot','histplot']
for plot in plot_kind:
  print('')
  numerical_plot(loan_dataset,plot,numerical_columns)

"""# **5. Data Preprocessing**

**5.1 Filling null Values**

**Categorical columns are filled with mode.**

**Numerical columns are filled with mean**
"""

loan_dataset.isnull().sum()

def fill_missing_values(df, columns):
    for column in columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    return df

columns = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
loan_dataset = fill_missing_values(loan_dataset, columns)

loan_dataset['LoanAmount'].fillna(loan_dataset['LoanAmount'].mean(),inplace=True)

loan_dataset.isnull().sum()

# drop Loan_ID column because is not necessary
loan_dataset = loan_dataset.drop('Loan_ID',axis = 1)

"""**5.2 Label Encoding**"""

labels_to_encode = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
copy = loan_dataset.copy()
copy2 = loan_dataset.copy()

def labelencoder(df,columns):
  encoder = LabelEncoder()
  for column in columns:
    df[column] = encoder.fit_transform(df[column])
  return df

new_dataset = labelencoder(loan_dataset,labels_to_encode)
new_dataset.head()

new_dataset['Dependents'].value_counts()

# replacing the value of 3+ to 4
new_dataset = loan_dataset.replace(to_replace= '3+', value=4)
copy = copy.replace(to_replace= '3+', value=4)

new_dataset['Dependents'].value_counts()

new_dataset['Loan_Status'].value_counts()

"""**5.3 Outliers Removal using Inter quaterline range**"""

q1 = new_dataset.quantile(0.25)
q3 = new_dataset.quantile(0.75)
iqr = q3-q1
lower_bound = q1 - (1.5*iqr)
upper_bound = q3 + (1.5*iqr)

new_dataset = new_dataset[~((new_dataset<lower_bound)| (new_dataset>upper_bound)).any(axis=1)]

# seperate features and target
features = new_dataset.iloc[:,:-1]
target = new_dataset['Loan_Status']

target.value_counts()

"""**Distribution of target column**

**1--> 183**

**2--> 37**

Our dataset is imballance so we use **SMOTE** (Synthetic Minority Over-sampling Technique).

It is a resampling technique that creates new synthetic samples from the minority class rather than oversampling with replacement
"""

sm = SMOTE()
features,target = sm.fit_resample(features,target)

target.value_counts().plot(kind='bar',title='Loan_Status')
# balanced data

"""**MINMAX SCALLING**"""

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

"""**Train Test Split**"""

train_data,test_data,train_labels,test_labels = train_test_split(features,target,test_size=0.2,random_state=0)

print('Shape of Original dataset: ',features.shape)
print('Shape of Train dataset: ',train_data.shape)
print('Shape of Test dataset: ',test_data.shape)

"""# **6. Modeling**

**Base model Logistic Regression**
"""

log_reg = LogisticRegression()
log_reg.fit(train_data,train_labels)

# prediction on Testing data
log_prediction = log_reg.predict(test_data)

# accuracy score
log_accuracy = accuracy_score(log_prediction,test_labels)
log_accuracy

# confusion matrix
log_conf = confusion_matrix(log_prediction,test_labels)
log_conf

# heatmap
sns.heatmap(log_conf,annot=True)
plt.show()

models = [LogisticRegression(max_iter=1000),SVC(kernel='linear'),KNeighborsClassifier(),RandomForestClassifier(n_estimators=25),XGBClassifier(),DecisionTreeClassifier(max_leaf_nodes=6)]
model_results = []
def best_model(model_list):
  for model in model_list:
    model.fit(train_data,train_labels)
    prediction = model.predict(test_data)
    accuracy = accuracy_score(prediction,test_labels)
    formated_answer = round(accuracy*100,2)
    model_results.append({
        'model Name':str(model),
        'Model Accuracy Score':formated_answer
    })
  return pd.DataFrame(model_results).sort_values(by='Model Accuracy Score',ascending=False)

    


best_model(models)


