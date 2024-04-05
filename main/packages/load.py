import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline

import statsmodels.tsa.stattools as ts
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

train_test_split_date = '2015-12-31'

def load_excel(file_path: str, name: str, sheet_name = 'Sheet 1', skiprows = 8, subset = False) -> pd.Series: 
    """
    Load Excel file containing HICP (Harmonized Index of Consumer Prices) all and 4 sub-group, clean up the data.
    
    Parameters:
    - file_path (str): The path to the Excel file.
    - name (str): Name of the series, to distinguish between different series.
    - sheet_name (str, optional): Name of the Excel sheet to load. Default is 'Sheet 1'.
    - skiprows (int, optional): Number of rows to skip from the beginning of the Excel sheet. Default is 8.
    - subset (bool, optional): If True, handle HICP subset, else handle HICP for all.
    
    Returns:
    - pd.Series: A pandas Series containing the cleaned up data.

    """
    data = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
    if subset: 
        data.dropna(axis = 1, how= 'all', inplace=True)
    df = data.iloc[1, :].to_frame().reset_index()
    df.columns = df.iloc[0]
    df = df[1:]
    df.rename(columns= {'TIME': 'date', 'Germany': name}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'] + pd.offsets.MonthEnd(0)
    df.dropna(subset=['date'], inplace=True)
    print(f"Data length: {df.shape[0]} rows from {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
    df.set_index('date', inplace=True)
    df = df.asfreq('M')
    df[df.columns[0]] = df.loc[:, name].astype('float')
    print(df.head(5))
    return df

def data_viz(df: pd.Series, title = None, add_line = False):
    """
    Lineplot time series.

    Parameters:
    - df (pd.Series): time series data.
    - title (str, optional): title of plot.
    - add_line (bool, optional): if True, add a horizonal line y = 0

    """
    plt.figure(figsize = (12, 4))
    if add_line:
        plt.plot(df,linewidth = 3)
        plt.axhline(y = 0)
    else:
        plt.plot(df)
    plt.title(title)

    plt.show()

def transform_yoy_rate(df: pd.DataFrame) -> pd.Series:
    """
    Transform HICP into year-on-year inflation rate (for HICP all and 4 sub categories)

    """
    df.loc[:, 'last_y'] = df.iloc[:, 0].shift(12)
    df.loc[:, 'yoy_rate'] = (df.iloc[:, 0]/df.loc[:, 'last_y'] - 1) * 100
    df = df.dropna(subset='yoy_rate')
    return df['yoy_rate']

def plots_acf_pacf(data: pd.Series, lags=None):
    """
    Plot ACF and PACF for time series.

    Parameters:
    - data (pd.Series): time series
    - lags (int, optional): maximum lags
    """
    plt.figure(figsize=(15, 4))
    layout = (1, 2)
    acf  = plt.subplot2grid(layout, (0, 0))
    pacf = plt.subplot2grid(layout, (0, 1))
    
    plot_acf(data, lags=lags, ax=acf, zero=False)
    plot_pacf(data, lags=lags, ax=pacf, zero = False)
    sns.despine()
    plt.tight_layout()

def dftest(timeseries):
    """
    Conduct ADF test for timeseries.
    """
    dftest = ts.adfuller(timeseries,) #call function adfuller 
    dfoutput = pd.Series(dftest[0:4],  
                         index=['Test Statistic','p-value','Lags Used','Observations Used'])
    # display first 4 values with its name
    print(dfoutput)

    if dfoutput['p-value'] < 0.05:
        print('Time series is stationary!')
    else:
        print('Time series is not stationary!')

class RecursiveForecast():
    def __init__(self, X: pd.DataFrame, y: pd.Series, max_horizon: int, model, hyperparameter) -> None:
        self.X = X
        self.y = y
        self.split_date = train_test_split_date
        self.max_horizon = max_horizon
        self.model = model
        self.hyperparameter = hyperparameter

        print(f"Train test split date: {self.split_date}, model {self.model}, hyperparameter: {self.hyperparameter}")

    def train_test_split(self): 
        """
        Split the train and test set based on the predetermined date

        """
        X_train = self.X[self.X.index < self.split_date] # as yt = f(Xt-1)
        X_test = self.X[self.X.index >= self.split_date].iloc[:-1, :]
        y_train, y_test = self.y[self.y.index <= self.split_date][1:], self.y[self.y.index > self.split_date]
        N, T = len(X_train), len(X_test)

        print(f'Training predictor period: {X_train.index[0]} to {X_train.index[-1]}')
        print(f'Training dependent variable period: {y_train.index[0]} to {y_train.index[-1]}')
        print(f'Test predictor period: {X_test.index[0]} to {X_test.index[-1]}')
        print(f'Test dependent variable period: {y_test.index[0]} to {y_test.index[-1]}')
        print('--------------------------------------------')

        return N, T, y_test
    
    def create_forecast_df(self, N: int) -> pd.DataFrame:
        """
        Create the forecast DataFrame based on the horizons
        Note: the first period in this df = last X_train - max_horizon + 1 so the furtherest forecast = first day of y_test
        Here: 2015-10 with 3 horizons => h_3 = 2016-01
        We'll chop it down later
        """
        column_names = [f"{self.model.__name__}_h_{i}" for i in range(1, self.max_horizon + 1)]
        index = self.X.iloc[N - self.max_horizon +1:, :].index

        # Create a DataFrame:
        init_forecast_df = pd.DataFrame(columns=column_names, index=index)
        
        return init_forecast_df
    

    def generate_forecast(self, forecast_df: pd.DataFrame, N: int, T: int) -> pd.DataFrame:

        for i in range(1, T + self.max_horizon): #
            X_train = self.X.iloc[:N - self.max_horizon + i +1, :]
            y_train = self.y[1:N - self.max_horizon + i + 2]
            # y_t = f(Xt-1)

            X_test = self.X.iloc[N - self.max_horizon +1+ i: N + 1+ i]
            # forecast next 3 period as horizon 1, 2, 3
            print(f'Predictor training period: {X_train.index[0]} to {X_train.index[-1]}')
            print(f'Forecast target period: {y_train.index[0]} to {y_train.index[-1]}')
            print(f'Predictor test period: {X_test.index}')

            model = self.model(self.hyperparameter)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test).ravel()
            print(f'Forecast: ')
            print(y_pred)
            if len(y_pred) < 3:
            # Pad with zeros to make it a length of 3
                y_pred = np.pad(y_pred, (0, 3 - len(y_pred)), 'constant')
            forecast_df.iloc[i-1, :] = y_pred

            print('-------------------------------------------------------')
        return forecast_df

    @staticmethod
    def chop_forecast_to_fit(forecast_df, y_test):
        """
        Chop the forecast to fit
        """
        forecast_result = pd.DataFrame(columns=forecast_df.columns, index= y_test[y_test.index < '2023-01-31'].index)
        forecast_result.iloc[:, 0] = forecast_df.iloc[2:-1, 0]
        forecast_result.iloc[:, 1] = forecast_df.iloc[1:-2, 1]
        forecast_result.iloc[:, 2] = forecast_df.iloc[:-3, 2]
        forecast_result.reset_index(drop=True, inplace=True)
        return forecast_result
    
    def concat_forecast(self):
        """
        Put all functions above into 1 pipeline
        """
        N, T, y_test = self.train_test_split()
        init_forecast_df = self.create_forecast_df(N = N)
        forecast_df = self.generate_forecast(init_forecast_df, N = N, T = T)
        final_forecast = self.chop_forecast_to_fit(forecast_df, y_test=y_test)
        return final_forecast

def save_forecast(forecast_result_df, cat_file_path, category = None):
    """
    Save the forecast into main file for comparision later
    - If there's no file created -> create file and save it, 
    - Else: import main file as a DataFrame, 
    if there's not yet results about that specific method, add it in
    otherwiser overwrite new results into dataframe.
    """
    if category is not None:
        for col in forecast_result_df.columns:
            col = col + category
    else: 
        pass
    if not os.path.isfile(cat_file_path):
    # File doesn't exist, create it and save the DataFrame
        forecast_result_df.to_csv(cat_file_path, index=False)
        print(f"CSV file '{cat_file_path}' created and DataFrame saved.")
    else:
        dataframe = pd.read_csv(cat_file_path)
        missing_columns = [col for col in forecast_result_df.columns if col not in dataframe.columns]

        if not missing_columns:
            dataframe.drop(columns=forecast_result_df.columns, inplace=True)
        concat_df = pd.concat([dataframe, forecast_result_df], axis= 1)
        concat_df.to_csv(cat_file_path, index=False)


    
### not done yet:
### Step 1: Get all necessary data:

def import_data_all(hicp_all_path: str, hicp_class_path: str, hicp_cat_path: str):
    """
    Import all necessary data
    
    """
    HICP_monthly = pd.read_csv(hicp_all_path)
    HICP_monthly['date'] = pd.to_datetime(HICP_monthly['date'])
    HICP_monthly.set_index('date', inplace=True)

    HICP_class = pd.read_excel(hicp_class_path, sheet_name='COICOP_class')
    HICP_class.index = ['Group 1', 'Group 2', 'Group 3', 'Group 4'] 

    HICP_cat = pd.read_csv(hicp_cat_path)
    HICP_cat['date'] = pd.to_datetime(HICP_cat['date'])
    HICP_cat.set_index('date', inplace=True)

    return HICP_monthly, HICP_class, HICP_cat


def split_into_category(category, HICP_class, HICP_monthly):
    """
    Take the subset of that specific categories
    """
    if category == 'Food':
        cat_col = HICP_class.loc[:, HICP_class.iloc[0, :] == category].columns
    else:
        cat_col = HICP_class.loc[:, HICP_class.iloc[1, :] == category].columns

    print(f'Number of items in {category} group: ', len(cat_col))

    cat_df = HICP_monthly[cat_col]
    cat_df.fillna(0, inplace=True)
    return cat_df

def split_train_set(cat_df, HICP_cat,h ,train_test_split_date = train_test_split_date):
    """
    Get the training set for hyperparameter tuning
    """
    X_cat_train = cat_df[cat_df.index <= train_test_split_date][:-h]
    y_cat_train = HICP_cat[HICP_cat.index <= train_test_split_date][h:]
    print(f'Horizon: {h}')
    print(f'Training predictor period: {X_cat_train.index[0]} to {X_cat_train.index[-1]}')
    print(f'Training dependent variable period: {y_cat_train.index[0]} to {y_cat_train.index[-1]}')
    return X_cat_train, y_cat_train


# Step 2: Hyperparameter tuning:

# 2.1. Ridge and Lasso, manual fine tuning:

def tuning_gridsearchcv(reg, grid_space, X_train, y_train, cv = 6, test_size = 12, scoring = 'neg_root_mean_squared_error'):
    """
    Tuning using GridSearchCV
    for Ridge and Lasso
    """

    tscv = TimeSeriesSplit(n_splits= cv, test_size= test_size)

    pipe = Pipeline(
        [
            ("scaling", StandardScaler()),
            ("regression", reg),
        ])

    param_grid = {
        'regression__alpha': grid_space
    }

    grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid, cv = tscv, scoring= scoring)
    grid.fit(X_train, y_train)

    return grid.best_params_

# 2.2. Tree-based models:

