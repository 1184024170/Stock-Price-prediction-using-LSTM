# Stock Price Predict on historical data using LSTM

# Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader.data as web
import quandl
#from datetime import datetime
seed = 2018
np.random.seed(seed)   # fix random seed for reproducibility

# Download stock price historical data 
data= quandl.get("EOD/AAPL", authtoken="FqZ3KgVJPNCQG6dy7hst")
data.head()
data.shape
data.columns


# Plot the graph
plt.figure(figsize=(10,8))
plt.plot(data['Adj_Close'], label='Close Price history')


# Realtive strength Index
def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

data["RSI-14"] = rsiFunc(data["Adj_Close"], n=14)

# Moving Average Convergence/Divergence (MACD)
data["12-ema"]= data["Adj_Close"].ewm(span=12, min_periods = 11).mean().fillna(method="bfill")
data["26-ema"] = data["Adj_Close"].ewm(span=26, min_periods = 25).mean().fillna(method="bfill")
data["MACD"] = data["12-ema"]-data["26-ema"].fillna(method="bfill")

# Simple Moving Average (SMA)
data["SMA"] = data["Adj_Close"].rolling(window=20, min_periods=19).mean().fillna(method="bfill")

# Bollinger Band
data["rolling_mean"] = data["Adj_Close"].rolling(21).mean().fillna(method="bfill")
data["rolling_std"] = data["Adj_Close"].rolling(21).std().fillna(method="bfill")
data["bollinger_high"] = data["rolling_mean"]+(data["rolling_std"]*2).fillna(method="bfill")
data["bollinger_low"] = data["rolling_mean"]-(data["rolling_std"]*2).fillna(method="bfill")

# move the Close column at the last position
Adj_Close = data.pop('Adj_Close')
data["Adj_Close"] = Adj_Close
new_data = data.copy()

# Check the missing values
print("Missing values: ", data.isnull().sum())
# No missing values


# Plot the correlation graph 
import seaborn as sns
corelation = data.corr()
sns.heatmap(corelation, 
        xticklabels=corelation.columns,
        yticklabels=corelation.columns)


# Normalize the data
from sklearn.preprocessing import  MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
for i in data.columns:
    data[i] = scaler.fit_transform(data[i].values.reshape(-1,1))
print(data)

# Split into training and testing dataset
train = data[:int(len(data)*0.9)].as_matrix()
test = data[len(train)+1:]
print(train.shape, test.shape)

# Convert dataset into X_train, y_train
X_train=[]
y_train=[]
for i in range(60, len(train)):
    X_train.append(train[i-60:i,:-1])
    y_train.append(train[i,-1])
   
X_train, y_train = np.array(X_train), np.array(y_train)


# Reshape the X_train as per tensorflow compatibility
X_train  = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

# Create the LSTM model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
adam = Adam()

sequence_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), kernel_initializer=initializers.glorot_normal(seed=seed), recurrent_initializer=initializers.Orthogonal(gain=1.0, seed=seed)))(sequence_input)
x = Dropout(0.2)(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])
conc = BatchNormalization()(conc)
conc = Dropout(0.2)(conc)
conc = Dense(64, activation="relu")(conc)
conc = Dropout(0.2)(conc)
sequence_output = Dense(1, activation="sigmoid")(conc)
model = Model(inputs=sequence_input, outputs= sequence_output)
model.compile(loss='mae', optimizer=adam)
print(model.summary())

# Set the callbacks
from keras.callbacks import EarlyStopping
early_stopping = early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=2, mode='auto')

# Fit the model
history = model.fit(X_train, y_train, batch_size=512, epochs=100, validation_split=0.2, verbose=2, callbacks=[early_stopping])

model.save("stock_200_epochs.h5")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# predicting the remaining values using past 60 from the train data
inputs = data[len(data)-len(test)-60:].as_matrix()

X_test = []
y_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, :-1])
    y_test.append(inputs[i,-1])

X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape the X_test as per tensorflow compatibility
X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))  
test_predict = model.predict(X_test)


# calculate root mean squared error
from sklearn.metrics import mean_squared_error
import math
rms = math.sqrt(mean_squared_error(y_test, test_predict))
rms

# plot the graph to see the results
train_graph = data[:int(len(data)*0.9)]
test_graph = data[len(train)+1:]
test_graph["Predictions"] = test_predict
#plt.plot(train_graph['Close'])
plt.plot(test_graph[['Predictions']])
plt.plot(test_graph[['Adj_Close']])
plt.title('Testing vs Predictions')
plt.ylabel('Normalized price')
plt.xlabel('Period')
plt.legend(['Pred', 'Test'], loc='upper left')