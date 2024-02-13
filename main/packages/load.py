import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.tsa.stattools as ts
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf



def load_excel(file_path, name, sheet_name = 'Sheet 1', skiprows = 8):
    """
    Load excel file of HICP all-item and 4 sub-groups
    """
    data = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
    df = data.iloc[1, :].to_frame().reset_index()
    df.columns = df.iloc[0]
    df = df[1:]
    df.rename(columns= {'TIME': 'date', 'Germany': name}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Data length: {df.shape[0]} rows from {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
    df.set_index('date', inplace=True)
    df = df.asfreq('MS')
    df[df.columns[0]] = df.loc[:, name].astype('float')
    print(df.head(5))
    return df
    

def data_viz(df, title):
    """
    Visualising time series
    """
    plt.figure(figsize = (12, 4))
    plt.plot(df)
    plt.title(title)
    plt.show()

def transform_yoy_rate(df):
    """
    Transform HICP into year-on-year inflation rate
    """
    df.loc[:, 'last_y'] = df.iloc[:, 0].shift(12)
    df.loc[:, 'yoy_rate'] = (df.iloc[:, 0]/df.loc[:, 'last_y'] - 1) * 100
    df = df.dropna(subset='yoy_rate')
    return df['yoy_rate']

def plots(data, lags=None):
    plt.figure(figsize=(15, 4))
    layout = (1, 2)
    acf  = plt.subplot2grid(layout, (0, 0))
    pacf = plt.subplot2grid(layout, (0, 1))
    
    plot_acf(data, lags=lags, ax=acf, zero=False)
    plot_pacf(data, lags=lags, ax=pacf, zero = False)
    sns.despine()
    plt.tight_layout()

def dftest(timeseries):
    dftest = ts.adfuller(timeseries,) #call function adfuller 
    dfoutput = pd.Series(dftest[0:4],  
                         index=['Test Statistic','p-value','Lags Used','Observations Used'])
    # display first 4 values with its name
    print(dfoutput)

    if dfoutput['p-value'] < 0.05:
        print('Time series is stationary!')
    else:
        print('Time series is not stationary!')
