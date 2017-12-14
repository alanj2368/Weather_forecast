# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:57:52 2017

@author: ALAN JACOB
"""

# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set = pd.read_csv('train.csv')
training_set = training_set.iloc[:,3:4].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the ouputs
X_train = training_set[0:15884]
y_train = training_set[1:15885]

# Reshaping
X_train = np.reshape(X_train, (15884, 1, 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None,1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
test_set = pd.read_csv('test.csv')
real_stock_price = test_set.iloc[:,3:4].values

# Getting the predicted stock price of 2017
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (89 ,1 , 1))
predicted_weather = regressor.predict(inputs)
predicted_weather = sc.inverse_transform(predicted_weather)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real weather')
plt.plot(predicted_weather, color = 'blue', label = 'Predicted weather')
plt.title('weather prediction')
plt.xlabel('Time')
plt.ylabel('weather')
plt.legend()
plt.show()




real_weather = pd.read_csv('test.csv')
real_weather = real_weather.iloc[:,3:4].values

# Getting the predicted stock price of 2012 - 2016
predicted_weather = regressor.predict(inputs)
predicted_weather = sc.inverse_transform(predicted_weather)

# Visualising the results
plt.plot(real_weather, color = 'red', label = 'Real weather')
plt.plot(predicted_weather, color = 'blue', label = 'Predicted weather')
plt.title('weather Prediction')
plt.xlabel('Time')
plt.ylabel('weather')
plt.legend()
plt.show()

# Part 4 - Evaluating the RNN

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_weather, predicted_weather))
