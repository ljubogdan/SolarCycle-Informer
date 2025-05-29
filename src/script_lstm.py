DATAPATH = 'data/Sunspots.csv'

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(DATAPATH)

# print(data.info()) # int64, object, float64

sunspots = data.iloc[:, -1] # the last column is the sunspots data
dates = data.iloc[:, -2] # the second column is the dates
# print(sunspots.values)

"""

plt.figure(figsize=(20, 6))
plt.plot(dates, sunspots)
plt.plot(sunspots[:72])    # from 1749 to 1755 (which is the new solar cycle) 
plt.plot(sunspots[72:72+132])  # first cycle
plt.plot(sunspots[-13:])  # current cycle
plt.ylabel(data.columns[-1])  
plt.xlabel(data.columns[-2])
plt.title('Sunspot numbers over time')
plt.xticks(ticks=np.arange(0, len(dates), 150), labels=dates.iloc[::150], rotation=45)
plt.legend(['All data', '1749-1755', 'First cycle', 'Current cycle'])
plt.tight_layout()
plt.show()

"""

"""

plt.figure(figsize=(20, 6))
plt.plot(range(72, len(sunspots)), sunspots[72:], label='Sunspots')
plt.title('Sunspot numbers from 1755 - 2019')
plt.xlim(72, len(sunspots) - 12)

# calculate year labels for every 11th year starting from 1755
start_year = 1755
num_years = (len(sunspots) - 72) // 12
years = [start_year + 11 * i for i in range((num_years // 11) + 1)]
xticks = [72 + 12 * 11 * i for i in range(len(years))]

plt.xticks(xticks, years, rotation=45)
plt.xlabel('Year')
plt.ylabel('Sunspot Number')
plt.legend()
plt.tight_layout()
plt.show()

"""

# RNN has problem with long sequences, so we will use LSTM

