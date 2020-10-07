import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV



sns.set_style("dark")

df = pd.read_csv("project_train.csv") # Load datasets
df_test = pd.read_csv("project_test.csv") # Load datasets
df = df.sample(frac=1, random_state = 1) # Shuffle data, set seed for reproducibility

print(df.head())

# Split dataset into training and test sets
###################################################
train_data = df[0:int(round(0.8*len(df)))]
test_data = df[int(round(0.8*len(df))):len(df)+1]

train_labels = train_data['Label']
test_labels = test_data['Label']

train_data = train_data.iloc[:,0:11]
test_data = test_data.iloc[:,0:11]
###################################################

## Regular logistic regression

logisticRegr = LogisticRegression(max_iter = 500) # Initialize regression model
logisticRegr.fit(train_data, train_labels)
#predictions = logisticRegr.predict(test_data)
test_score = logisticRegr.score(test_data,test_labels)
print('Test score without CV = ', str(test_score))

## Logistic regression with cross-validation

CVLogReg = LogisticRegressionCV(cv = 3, max_iter = 500, random_state=0)
CVLogReg.fit(train_data,train_labels)
CV_test_score = CVLogReg.score(test_data,test_labels)
print('Test score with CV = ', str(CV_test_score))

## Perform same experiments again, but with either 'acousticness' or 'instrumentalness' removed
train_data_slim = train_data.drop('acousticness', axis = 1)
test_data_slim = test_data.drop('acousticness', axis = 1)

CVLogRegSlim = LogisticRegressionCV(cv = 5, max_iter = 500, random_state=0)
CVLogRegSlim.fit(train_data_slim,train_labels)
slim_test_score = CVLogRegSlim.score(test_data_slim,test_labels)
print('Test score with CV with dropped variables = ', str(slim_test_score))
