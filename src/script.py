'''

    For this model, we are using Informer model (Transformer-based model for time series forecasting).
    Seasolanity is handled by the model itself, so we do not need to add any additional features.
    Trend is also handled by the model itself.

    We tried to use the LSTM model (RNN-based model for time series forecasting), but it did not perform well.
    We tried to use the ARIMA model, no results were obtained.
    We tried to use the SARIMA model (Seasonal ARIMA-based model for time series forecasting), no results also...

    All thee models were not able to capture the seasonality and trend in the data.

    We are working with sunspot data, which is not linear and has a complex seasonality and trend.
    
'''

# INFORMER MODEL

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import statsmodels.api as sm

'''
    === COLUMNS IN SUNSPOT MONTLY TOTAL DATA ===

    year
    month
    date_in_fraction (e.g. 2023-01-01 is 2023.0, 2023-01-02 is 2023.003, etc.)
    total_sunspots
    deviation 
    number_of_observations 
    definitive/provisional 

    === COLUMNS IN SUNSPOT MONTHY SMOOTHED DATA ===

    (identical to the monthly total data, but with a 13-month moving average applied)

    === COLUMNS IN HEMISPHERIC MONTHLY TOTAL DATA ===

    year
    month
    date_in_fraction
    north+south (total sunspots in both hemispheres)
    north (total sunspots in northern hemisphere)
    south (total sunspots in southern hemisphere)
    deviation 
    deviation_north 
    deviation_south 
    observations 
    observations_north 
    observations_south 
    definitive/provisional 

    === COLUMNS IN HEMISPHERIC MONTHLY SMOOTHED DATA: ===

    (identical to the monthly total data, but with a 13-month moving average applied)

    WE HAVE DATA FROM JANARY 1749 TO APRIL 2025
    FIRST AND LAST 6 MONTHS OF DATA IN SMOOTHED ARE MISSING (-1 in column indicates missing data) 

    WE HAVE HEMISPHERIC DATA FROM JANARY 1992 TO APRIL 2025



'''

# Plotting 4 graphs in one figure, one above the other
# monthly total data, monthly smoothed data, hemispheric monthly total data, hemispheric monthly smoothed data
# where in hemispheric data we plot 2 lines, one for north and one for south

def plot_sunspot_data(df_total, df_smoothed, df_hemispheric_total, df_hemispheric_smoothed):
    fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)

    # Monthly Total Data
    axs[0].plot(df_total['date_in_fraction'], df_total['total_sunspots'], label='Total Sunspots', color='blue')
    axs[0].set_title('Monthly Total Sunspots')
    axs[0].set_ylabel('Total Sunspots')
    axs[0].legend()
    axs[0].grid()

    # Monthly Smoothed Data
    axs[1].plot(df_smoothed['date_in_fraction'], df_smoothed['total_sunspots'], label='Smoothed Total Sunspots', color='orange')
    axs[1].set_title('Monthly Smoothed Sunspots')
    axs[1].set_ylabel('Smoothed Total Sunspots')
    axs[1].legend()
    axs[1].grid()

    # Hemispheric Monthly Total Data
    axs[2].plot(df_hemispheric_total['date_in_fraction'], df_hemispheric_total['north+south'], label='North + South', color='green')
    axs[2].plot(df_hemispheric_total['date_in_fraction'], df_hemispheric_total['north'], label='North', color='red')
    axs[2].plot(df_hemispheric_total['date_in_fraction'], df_hemispheric_total['south'], label='South', color='purple')
    axs[2].set_title('Hemispheric Monthly Total Sunspots')
    axs[2].set_ylabel('Total Sunspots (North + South)')
    axs[2].legend()
    axs[2].grid()

    # Hemispheric Monthly Smoothed Data
    axs[3].plot(df_hemispheric_smoothed['date_in_fraction'], df_hemispheric_smoothed['north+south'], label='Smoothed North + South', color='green')
    axs[3].plot(df_hemispheric_smoothed['date_in_fraction'], df_hemispheric_smoothed['north'], label='Smoothed North', color='red')
    axs[3].plot(df_hemispheric_smoothed['date_in_fraction'], df_hemispheric_smoothed['south'], label='Smoothed South', color='purple')
    axs[3].set_title('Hemispheric Monthly Smoothed Sunspots')
    axs[3].set_ylabel('Smoothed Total Sunspots (North + South)')
    axs[3].legend()
    axs[3].grid()

    plt.xlabel('Date in Fraction')
    plt.tight_layout()
    plt.show()

# Load the data

def load_data():
    # Load the data from CSV files
    df_total = pd.read_csv('../data/SN_m_tot_V2.0.csv')
    df_smoothed = pd.read_csv('../data/SN_ms_tot_V2.0.csv')
    df_hemispheric_total = pd.read_csv('../data/SN_m_hem_V2.0.csv')
    df_hemispheric_smoothed = pd.read_csv('../data/SN_ms_hem_V2.0.csv')

    # Convert date_in_fraction to datetime for better plotting
    df_total['date_in_fraction'] = pd.to_datetime(df_total['date_in_fraction'], format='%Y.%m')
    df_smoothed['date_in_fraction'] = pd.to_datetime(df_smoothed['date_in_fraction'], format='%Y.%m')
    df_hemispheric_total['date_in_fraction'] = pd.to_datetime(df_hemispheric_total['date_in_fraction'], format='%Y.%m')
    df_hemispheric_smoothed['date_in_fraction'] = pd.to_datetime(df_hemispheric_smoothed['date_in_fraction'], format='%Y.%m')

    return df_total, df_smoothed, df_hemispheric_total, df_hemispheric_smoothed

def main():
    # Load the data
    df_total, df_smoothed, df_hemispheric_total, df_hemispheric_smoothed = load_data()

    # Plot the data
    plot_sunspot_data(df_total, df_smoothed, df_hemispheric_total, df_hemispheric_smoothed)

if __name__ == "__main__":
    main()
