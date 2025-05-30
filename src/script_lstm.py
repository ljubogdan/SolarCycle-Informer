"""
    Script used for prediction of sunspot numbers using LSTM.
    The script loads the dataset, preprocesses it, and visualizes the data.

    Author: Bogdan Ljubinković
    Date: May 2025
"""

DATA_PATH = '/kaggle/input/datasunspots/SN_m_tot_V2.0.csv'

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# loading dataset
sunspots = pd.read_csv(DATA_PATH, sep=';')

columns = ["year", "month", "decimal_year", "number", "deviation", "observations", "flag"]

print(sunspots.info())

"""
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   year          3316 non-null   int64  
 1   month         3316 non-null   int64  
 2   decimal_year  3316 non-null   float64
 3   number        3316 non-null   float64
 4   deviation     3316 non-null   float64
 5   observations  3316 non-null   int64  
 6   flag          3316 non-null   int64  
"""

# eliminate unnecessary columns
sunspots.drop(columns=['flag', 'deviation', 'observations'], inplace=True)

print(sunspots.info())

"""
Data columns (total 4 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   year          3316 non-null   int64  
 1   month         3316 non-null   int64  
 2   decimal_year  3316 non-null   float64
 3   number        3316 non-null   float64
dtypes: float64(2), int64(2)
memory usage: 103.8 KB
None
"""

# merge year and month into a single date column
sunspots['date'] = pd.to_datetime(sunspots[['year', 'month']].assign(day=1))

# eliminate year, month, and decimal_year columns
sunspots.drop(columns=['year', 'month', 'decimal_year'], inplace=True)
print(sunspots.info())

"""
 #   Column  Non-Null Count  Dtype         
---  ------  --------------  -----         
 0   number  3316 non-null   float64       
 1   date    3316 non-null   datetime64[ns]
dtypes: datetime64[ns](1), float64(1)
memory usage: 51.9 KB
"""

# plotting the sunspot numbers over time

import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# plot graph, where first 77 data points are in red (first solar cycle)
# next 134 data points are in green (second solar cycle)
# and the last 50 data points are in orange (solar cycle 25)

def first_plot():
    plt.figure(figsize=(18, 6))
    plt.plot(sunspots['date'], sunspots['number'], label='Sunspot Number', color='blue')
    plt.plot(sunspots['date'][:77], sunspots['number'][:77], label='First Solar Cycle', color='red')
    plt.plot(sunspots['date'][77:210], sunspots['number'][77:210], label='Second Solar Cycle', color='green')
    plt.plot(sunspots['date'][-60:], sunspots['number'][-60:], label='Solar Cycle 25', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Sunspot Number')
    plt.title('Sunspot Number Over Time with Solar Cycles Highlighted')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(12))  # maximum 12 ticks on x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # format x-axis as Year-Month
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# first_plot()

# remove first 77 and last 60 data points (incomplete solar cycles)
sunspots = sunspots.iloc[77:-60].reset_index(drop=True)

# plot graph, where x axis shows first date and every 132th date after that till the end
new_dates = []
for i in range(0, len(sunspots), 132):
    new_dates.append(sunspots['date'].iloc[i])

def second_plot(new_dates):
    plt.figure(figsize=(18, 6))
    plt.plot(sunspots['date'], sunspots['number'], label='Sunspot Number', color='blue')
    plt.xticks(new_dates, rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Sunspot Number')
    plt.title('Sunspot Number Over Time with Selected Dates')
    
    # new_dates on x-axis

    new_dates = mdates.date2num(new_dates)  # convert to matplotlib date format

    plt.gca().xaxis.set_major_locator(ticker.FixedLocator(new_dates))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # format x-axis as Year-Month
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# second_plot(new_dates)

# plotting the acf
from statsmodels.graphics.tsaplots import plot_acf
def plot_acf_graph():
    fig = plt.figure(figsize=(16, 6))
    plot_acf(sunspots['number'], lags=30, alpha=0.05, title='Autocorrelation Function of Sunspot Numbers', ax=plt.gca())
    plt.tight_layout()
    plt.show()

# plot_acf_graph()

"""
    As we can see from the ACF graph after k=29 there is no autocorrelation
    Which means this is a time series
"""

# plotting the pacf
from statsmodels.graphics.tsaplots import plot_pacf
def plot_pacf_graph():
    fig = plt.figure(figsize=(16, 6))
    plot_pacf(sunspots['number'], lags=30, alpha=0.05, title='Partial Autocorrelation Function of Sunspot Numbers', ax=plt.gca())
    plt.tight_layout()
    plt.show()

# plot_pacf_graph()

"""
    As we can see from the PACF graph after k=7 there is no partial autocorrelation
    Currently sunspot numbers are not stationary
    Because we have a trend in the data, and sensonality
    PACF doesnt eliminate effect of trend and seasonality

    Additive or multiplicative decomposition of time series?

    We will use multiplicative decomposition...
    But STL requires additive model!!!

    Transform multiplicative model to additive model by taking logarithm of the data
"""

# removing all rows with number == 0
sunspots = sunspots[sunspots['number'] > 0].reset_index(drop=True)

sunspots['log_number'] = np.log10(sunspots['number'])

"""
sunspots['log_number'].plot(figsize=(18, 6), title='Logarithm of Sunspot Numbers', xlabel='Date', ylabel='Log(Sunspot Number)')
plt.tight_layout()
plt.show()
"""

# stl method for decomposition
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# STL decomposition of the logarithm of sunspot numbers
stl = STL(sunspots['log_number'], period=132).fit()

# Plotting components explicitly

"""
fig, axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
axes[0].plot(sunspots.index, sunspots['log_number'], label='Original', color='blue')
axes[0].legend(loc='upper left')

axes[1].plot(sunspots.index, stl.trend, label='Trend', color='orange')
axes[1].legend(loc='upper left')

axes[2].plot(sunspots.index, stl.seasonal, label='Seasonality', color='green')
axes[2].legend(loc='upper left')

axes[3].plot(sunspots.index, stl.resid, label='Residuals', color='red')
axes[3].legend(loc='upper left')

plt.tight_layout()
plt.show()
"""

"""
    As we can see there is trend and seasonality in the data
    Stationarity means that properties such as mean, variance and covariance
    do not change over time.
    For predictive models we need to have stationary time series.
"""

# Checking statioonarity using ADF test
from statsmodels.tsa.stattools import adfuller
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    # printing wether series is stationary or not
    if result[1] <= 0.05:
        print("Series is stationary")
    else:
        print("Series is not stationary")

# adf_test(sunspots['number'])

"""
    Apperantl series is stationary? - but why?
    ADF test is sometimes not reliable
    We will use KPSS test to check for stationarity ;)

    KPSS test is a statistical test that checks for stationarity in a time series.
    The null hypothesis of the KPSS test is that the time series is stationary.
"""

from statsmodels.tsa.stattools import kpss
def kpss_test(series):
    statistic, p_value, lags, critical_values = kpss(series, regression='c')
    print('KPSS Statistic:', statistic)
    print('p-value:', p_value)
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value}')
    
    # printing wether series is stationary or not
    if p_value <= 0.05:
        print("Series is not stationary")
    else:
        print("Series is stationary")

# kpss_test(sunspots['number'])

"""
    As we can see from the KPSS test, series is actually stationary
    For now, we will believe that series is stationary
    BUT we still see trend in data, and we need to remove it
    So we will differentiate the data to remove trend
"""

original = sunspots['number'].iloc[0]

# removing trend from the data
sunspots['differentiated'] = sunspots['number'].diff()
sunspots = sunspots.dropna(subset=['differentiated'])

# adf test again
# adf_test(sunspots['differentiated'])

# plotting the diff data

def plot_diff():
    plt.figure(figsize=(18, 6))
    plt.plot(sunspots['date'], sunspots['differentiated'], label='Differentiated Sunspot Number', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Differentiated Sunspot Number')
    plt.title('Differentiated Sunspot Number Over Time')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(12))  # maximum 12 ticks on x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # format x-axis as Year-Month
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# plot_diff()

# adf test again
# adf_test(sunspots['differentiated'])

"""
    As we can see, series is stationary now
    We can proceed with LSTM model
"""

# its time to prepare data for LSTM model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

scaler = MinMaxScaler()

# scaling the data
split_index = int(len(sunspots) * 0.8)
train_data = sunspots['differentiated'][:split_index]
test_data = sunspots['differentiated'][split_index:]

scaler.fit(train_data.values.reshape(-1, 1)) # very important

sunspots['scaled'] = np.nan
train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
test_scaled = scaler.transform(test_data.values.reshape(-1, 1)).flatten()

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 30  

X_train, y_train = create_sequences(train_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)

# reshaping data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# importing necessary libraries for LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation

model = Sequential()
model.add(LSTM(50, input_shape=(window_size, 1), return_sequences=True))
model.add(Activation('tanh'))

model.add(LSTM(100, return_sequences=True))
model.add(Activation('tanh'))
model.add(Dropout(0.3))

model.add(LSTM(100))
model.add(Activation('tanh'))

model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Activation('linear'))

model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()

epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=20, validation_split=0.2)

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])

y_predict = model.predict(X_test)
y_predict_orig = scaler.inverse_transform(y_predict.reshape(-1, 1)).flatten()
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

"""
    y_predict_orig and y_test_orig are differentiated values
    We need to add the last value of the training data to the first value of the test data
    to get the original values
"""



RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2)

plt.figure(figsize=(10,6))
plt.plot(sunspots['date'][split_index + window_size:], y_test_orig, label='True Values', color='blue')
plt.plot(sunspots['date'][split_index + window_size:], y_predict_orig, label='Predicted Values', color='red')
plt.xlabel('Date')
plt.ylabel('Sunspot Number')
plt.title('Sunspot Number Prediction using LSTM')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(12))  # maximum 12 ticks on x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # format x-axis as Year-Month
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


