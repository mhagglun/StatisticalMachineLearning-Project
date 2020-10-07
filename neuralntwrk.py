import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2

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

train_data = (train_data - train_data.mean())/train_data.std()
test_data = (test_data - test_data.mean())/test_data.std()

train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

train_labels = train_labels.to_numpy()
test_labels = test_labels.to_numpy()

# Scale dataset

###################################################

# define and fit the final model

initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

model = Sequential()
model.add(Dense(64, input_dim=11, activation='relu', kernel_initializer=initializer))
model.add(Dense(128, activation='relu', kernel_initializer=initializer))
model.add(Dense(8, activation='relu', kernel_initializer=initializer))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
history = model.fit(train_data, train_labels, epochs=50, batch_size = 1, validation_split = 0.1)
plt.plot(history.history['val_loss'])
plt.show()
predictions = model.predict_classes(test_data)
print(accuracy_score(test_labels,predictions))
