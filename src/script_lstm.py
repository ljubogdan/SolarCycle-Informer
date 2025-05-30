"""
    Script used for prediction of sunspot numbers using LSTM.
    The script loads the dataset, preprocesses it, and visualizes the data.

    Author: Bogdan Ljubinković
    Date: May 2025
"""

DATA_PATH = 'data/SN_m_tot_V2.0.csv'

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

# 











