import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
from tensorflow import keras
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

sns.set_style("dark")

# Import data
###################################################
df = pd.read_csv("project_train.csv") # Load datasets
df = df.sample(frac=1, random_state = 1) # Shuffle data, set seed for reproducibility
print(df.head())
###################################################


# Split dataset into training and test sets
###################################################
train_data = df[0:int(round(0.8*len(df)))]
test_data = df[int(round(0.8*len(df))):len(df)+1]

train_labels = train_data['Label']
test_labels = test_data['Label']

train_data = train_data.iloc[:,0:11]
test_data = test_data.iloc[:,0:11]

train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

train_labels = train_labels.to_numpy()
test_labels = test_labels.to_numpy()

###################################################

# define and fit the final model
model = Sequential()
model.add(Dense(64, input_dim=11, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(train_data, train_labels, epochs=200, verbose=0)

predictions = model.predict_classes(test_data)
print(accuracy_score(test_labels,predictions))
