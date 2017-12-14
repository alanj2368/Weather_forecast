import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
training_set = pd.read_csv('train.csv')
training_set = training_set.iloc[:, 3:4].values
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:15884]
y_train = training_set[1:15885]
X_train = np.reshape(X_train, (15884, 1, 1))
regressor = Sequential()
regressor.add(LSTM(units=4, activation='sigmoid', input_shape = (None, 1)))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, batch_size=16, epochs=600)
test_set = pd.read_csv('test.csv')
real_weather = test_set.iloc[:, 3:4].values
inputs = real_weather
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (89, 1, 1))
predicted_weather = regressor.predict(inputs)
predicted_weather = sc.inverse_transform(predicted_weather)
plt.plot(real_weather, color='red', label='Real weather')
plt.plot(predicted_weather, color='green', label='Predicted weather')
plt.title('weather prediction')
plt.xlabel('Time')
plt.ylabel('weather')
plt.legend()
plt.show()
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_weather, predicted_weather))
