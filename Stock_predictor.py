# Importing Necessary libraries 
import pandas as pd
import numpy as np 

# Data Visualisation 
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning Libraries  
import keras

# Uploading File to colab ( comment it out if not using colab )

#from google.colab import files
# files.upload()

# Loading the data into a pandas data frame
train_data = pd.read_csv('Google_Stock_Price_Train.csv')
test_data = pd.read_csv('Google_Stock_Price_Test.csv')
print(train_data.shape, test_data.shape)

# Displaying the first five rows
print(train_data.head())

# Plotting the 'Open Price' Data wrt time
plt.plot(train_data['Date'], train_data['Open'])
plt.xlabel('Date')
plt.ylabel('Open Stock Price Data')
plt.show()

# Creating a 2dNumpy array of all data of the open column
train_set = train_data[:, 1 : 2].values
print(type(train_set))

# Feature Scaling using the Min Max Scaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(train_set)
print(scaled_data)

# Creating a list of the numpy arrays containing the last 60 days of prices 
# for each individual price in y 
X_train = []
y_train = []
# window size = 60
for i in range(60, 1258):
    X_train.append(scaled_data[i - 60 : i, 0])
#     X_train = scaled_data[i-60 : i, 0]
#     y_train = scaled_data[i:i+1, 0]
    y_train.append(scaled_data[i, 0])
# converting to arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the data - used to add a new dimension to the data

'''Input shape: 3D tensor with shape (batch_size, timesteps, input_dim)'''

batch_size = X_train.shape[0]
timesteps = 60
no_of_predictors = 1 # input_dim
X_train = np.reshape(X_train, (batch_size, timesteps, no_of_predictors))
print(X_train.shape)
print(X_train)

''' BUILDING THE RNN '''

# Importing the necessary Keras libraries
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Building our regressor
regressor = Sequential()

# Adding the first LSTM layer and a dropout regularisation layer
regressor.add(LSTM(units = 60, 
                  return_sequences = True, 
                  input_shape = (X_train.shape[1], 1)))

# Dropout regularisation 
regressor.add(Dropout(rate = 0.2)) # 20% dropout

# Adding the second LSTM layer and a dropout regularisation 
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Adding the third LSTM layer and dropout regularisation 
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Adding the fourth LSTM layer and dropout regularisation 
regressor.add(LSTM(units = 60, return_sequences = False))
regressor.add(Dropout(rate = 0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling our RNN 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training Set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

''' Predicting the results on our test dataset''' 

# Getting the real Google stock price of 2017 from the test set
real_stock_price = test_data.iloc[:, 1:2].values
print(real_stock_price)

# Getting the predicted stock price 

# Concatenate the train and test data of open stock price
total_data = pd.concat((train_data['Open'], test_data['Open']), axis = 0)
inputs = total_data[len(total_data) - len(test_data) - 60:].values

# Reshaping the input array 
inputs = inputs.reshape(-1, 1)

# Scaling the input data
inputs = scaler.transform(inputs)

# Making our predictions 

X_test = []
for i in range(60, 80): 
    X_test.append(inputs[i - 60 : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test)

# Making predictions
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
predicted_stock_price

''' Visualising the predictions ''' 

plt.plot(predicted_stock_price, color = 'r', label = 'Predicted Stock Price (Google)')
plt.plot(real_stock_price, color = 'g', label = 'Real Stock Price (Google)')

plt.title('Stock Price Predictions (Google)')

plt.xlabel('Time Period')
plt.ylabel('Open Google Stock Price')
plt.legend()
plt.show()

# Measuring the Error
from sklearn.metrics import mean_squared_error, mean_absolute_error

mae_acc = mean_absolute_error(y_true = real_stock_price, y_pred = predicted_stock_price)
mse_acc = mean_squared_error(y_true = real_stock_price, y_pred = predicted_stock_price)
print("Mean Absolute Error: {}".format(mae_acc))
print("Mean Suared Error: {}".format(mse_acc))
