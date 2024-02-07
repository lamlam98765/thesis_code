import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    df.set_index('date', inplace=True)
    print(f"Data length: {df.shape[0]} rows from {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
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