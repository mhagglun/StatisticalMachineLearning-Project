import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression



sns.set_style("dark")

df = pd.read_csv("project_train.csv") # Load datasets
df = df.sample(frac=1, random_state = 1) # Shuffle data, set seed for reproducibility

print(df.head())


# Split dataset into training and test sets
train_data = df[0:int(round(0.7*len(df)))]
test_data = df[int(round(0.7*len(df))):len(df)+1]

train_labels = train_data['Label']
test_labels = test_data['Label']

train_data = train_data.iloc[:,0:11]
test_data = test_data.iloc[:,0:11]


logisticRegr = LogisticRegression(max_iter = 500) # Initialize regression model
logisticRegr.fit(train_data, train_labels)
#predictions = logisticRegr.predict(test_data)
test_score = logisticRegr.score(test_data,test_labels)
print(test_score)
