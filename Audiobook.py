import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection

dataset = pd.read_csv('13. Business case __ 1.1 Audiobooks_data.csv') #Read the dataset

# The data is unballanced with less than 15% of returned customers
target_one_dataset = dataset.loc[dataset.Targets == 1] #the Target tag of 1 represent the returned customers
target_zero_dataset = dataset.loc[dataset.Targets == 0]
target_zero_dataset = target_zero_dataset.sample(3*target_one_dataset.shape[0]) # take a subset of dataset with 0.25% 0 Target tag
balanced_dataset = pd.concat([target_one_dataset, target_zero_dataset])
balanced_dataset = balanced_dataset.drop('ID', axis=1)
X = balanced_dataset.iloc[:, :-1]
Y = balanced_dataset.iloc[:, -1]
X = preprocessing.scale(X)
x_train, X_test, y_train, Y_test = model_selection.train_test_split(X, Y, random_state=1, shuffle=True)
X_train, X_cross_validation, Y_train, Y_cross_validation = model_selection.train_test_split(x_train,
                                                y_train, test_size=0.1, shuffle=True) #split dataset to train, cross validation, and test sets

# define input, hidden layers, and output size
input_size = 10
output_size = 2
hidden_layer_size = 60

# a model with 3 hidden layers and relu activation function for all layers except last one shows the best result and also avoid overfitting
model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_layer_size, activation= 'relu'),
                             tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                             tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                             tf.keras.layers.Dense(output_size, activation= 'softmax')
])

print(X_train.shape, Y_train.shape, X_cross_validation.shape, Y_cross_validation.shape, X_test.shape, Y_test.shape)
model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
batch_size = 100
max_epochs = 100
early_stop = tf.keras.callbacks.EarlyStopping(patience= 3)
model.fit(X_train, Y_train, batch_size= batch_size, epochs = max_epochs,
          validation_data= (X_cross_validation, Y_cross_validation), verbose= 2, callbacks= [early_stop])

test_loss, test_accuracy= model.evaluate(X_test, Y_test)