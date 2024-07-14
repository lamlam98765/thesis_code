"""
General code used by all models, including importing and preprocessing data.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.tsa.stattools as ts
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

train_test_split_date = pd.to_datetime("2015-12-31")  # the date to split train-test set
max_X_date = pd.to_datetime("2022-12-31")


### Preprocessing data:
def load_excel(
    file_path: str, name: str, sheet_name="Sheet 1", skiprows=8, subset=False, verbose=1
) -> pd.Series:
    """
    Load Excel file containing HICP (Harmonized Index of Consumer Prices) all and 4 sub-group, clean up the data.

    Parameters:
    - file_path (str): The path to the Excel file.
    - name (str): Name of the series, to distinguish between different series.
    - sheet_name (str, optional): Name of the Excel sheet to load. Default is 'Sheet 1'.
    - skiprows (int, optional): Number of rows to skip from the beginning of the Excel sheet. Default is 8.
    - subset (bool, optional): If True, handle HICP subset, else handle HICP for all.
    - verbose (int, optional): If verbose == 1, print additional information.

    Returns:
    - pd.Series: A pandas Series containing the cleaned up data.

    """
    data = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
    if subset:
        data.dropna(axis=1, how="all", inplace=True)
    df = data.iloc[1, :].to_frame().reset_index()
    df.columns = df.iloc[0]
    df = df[1:]
    df.rename(columns={"TIME": "date", "Germany": name}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    df.dropna(subset=["date"], inplace=True)
    df.set_index("date", inplace=True)
    df = df.asfreq("M")
    df[df.columns[0]] = df.loc[:, name].astype("float")
    if verbose == 1:
        print(
            f"Data length: {df.shape[0]} rows from {df.iloc[0, 0]} to {df.iloc[-1, 0]}"
        )
        print(df.head(5))
    return df


def data_viz(df: pd.Series, title=None, add_line=False):
    """
    Lineplot time series.

    Parameters:
    - df (pd.Series): time series data.
    - title (str, optional): title of plot.
    - add_line (bool, optional): if True, add a horizonal line y = 0

    Return:
    None
    """
    plt.figure(figsize=(12, 4))
    if add_line:
        plt.plot(df, linewidth=3)
        plt.axhline(y=0)
    else:
        plt.plot(df)
    plt.title(title)

    plt.show()


def transform_yoy_rate(df: pd.DataFrame) -> pd.Series:
    """
    Transform HICP data into year-on-year inflation rate (for HICP all and 4 sub categories).

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the HICP data. The first column should represent the HICP values.

    Returns:
    - pd.Series: A Series containing the calculated year-on-year inflation rates.
    """
    # 1. Shift by 12 rows to obtain the value from the same month in the previous year.
    df.loc[:, "last_y"] = df.iloc[:, 0].shift(12)

    # 2. Compute the year-on-year inflation rate using the formula: ((current_value / last_year_value) - 1) * 100.
    df.loc[:, "yoy_rate"] = (df.iloc[:, 0] / df.loc[:, "last_y"] - 1) * 100

    # 3. Remove any NaN rows.
    df = df.dropna(subset="yoy_rate")
    return df["yoy_rate"]


def plots_acf_pacf(data: pd.Series, lags=None):
    """
    Plot ACF and PACF for time series.

    Parameters:
    - data (pd.Series): time series
    - lags (int, optional): maximum lags

    Return:
    None.
    """

    plt.figure(figsize=(15, 4))
    layout = (1, 2)
    acf = plt.subplot2grid(layout, (0, 0))
    pacf = plt.subplot2grid(layout, (0, 1))

    plot_acf(data, lags=lags, ax=acf, zero=False)
    plot_pacf(data, lags=lags, ax=pacf, zero=False)
    sns.despine()
    plt.tight_layout()


def dftest(timeseries: pd.Series):
    """
    Conducts the Augmented Dickey-Fuller (ADF) test to check for stationarity in a time series.

    Parameters:
    - timeseries (pd.Series): time series

    Returns:
    None

    The function prints the ADF test results and indicates if the time series is stationary based on the p-value.
    """

    dftest = ts.adfuller(
        timeseries,
    )  # call function adfuller
    dfoutput = pd.Series(
        dftest[0:4],
        index=["Test Statistic", "p-value", "Lags Used", "Observations Used"],
    )
    # display first 4 values with its name
    print(dfoutput)

    if dfoutput["p-value"] < 0.05:
        print("Time series is stationary!")
    else:
        print("Time series is not stationary!")


def save_forecast(forecast_result_df, cat_file_path):
    """
    Save the forecasts.
    Parameters:
    - forecast_result_df (pd.DataFrame): DataFrame containing the forecast results.
    - cat_file_path (str): Path to the file where results are stored.

    Returns:
    None

    This function performs the following steps:
    1. If the file does not exist, create it and save the forecast results.
    2. If the main file exists, load it as a DataFrame.
    3. If results for the specified method/category do not exist, add the new results.
    4. If results for the specified method/category exist, overwrite them with the new results.
    """

    if not os.path.isfile(cat_file_path):
        # File doesn't exist, create it and save the DataFrame
        forecast_result_df.to_csv(cat_file_path, index=False)
        print(f"CSV file '{cat_file_path}' created and DataFrame saved.")
    else:
        dataframe = pd.read_csv(cat_file_path)
        missing_columns = [
            col for col in forecast_result_df.columns if col not in dataframe.columns
        ]

        if not missing_columns:
            dataframe.drop(columns=forecast_result_df.columns, inplace=True)
        concat_df = pd.concat([dataframe, forecast_result_df], axis=1)
        concat_df.to_csv(cat_file_path, index=False)


### Import all necessary data:


def import_data_all(hicp_all_path: str, hicp_class_path: str, hicp_cat_path: str):
    """
    Import all necessary data, first step in forecasting.

    Parameters:
    - hicp_all_path (str): path to HICP all-item data.
    - hicp_class_path (str): path to HICP classification.
    - hicp_cat_path (str): path to a specific HICP category

    Return:
    Preprocessed HICP all-items, HICP classification, HICP of that category.
    """

    HICP_monthly = pd.read_csv(hicp_all_path)
    HICP_monthly["date"] = pd.to_datetime(HICP_monthly["date"])
    HICP_monthly.set_index("date", inplace=True)

    HICP_class = pd.read_excel(hicp_class_path, sheet_name="COICOP_class")
    HICP_class.index = ["Group 1", "Group 2", "Group 3", "Group 4"]

    HICP_cat = pd.read_csv(hicp_cat_path)
    HICP_cat["date"] = pd.to_datetime(HICP_cat["date"])
    HICP_cat.set_index("date", inplace=True)

    return HICP_monthly, HICP_class, HICP_cat


def split_into_category(category, HICP_class, HICP_monthly, fillna=True):
    """
    Extract a subset of data for a specific category from HICP data.

    Parameters:
    - category : str
        The name of the category to filter by.
    - HICP_class : pandas.DataFrame
        A DataFrame containing category classifications.
    - HICP_monthly : pandas.DataFrame
        A DataFrame containing the monthly HICP data.
    - fillna : bool, optional
        Whether to fill missing values with zero in the resulting subset DataFrame. The default is True.

    Returns:
    pandas.DataFrame
        A subset of `HICP_monthly` containing only the columns that belong to
        the specified category.

    Prints:
    Number of items in the specified category.

    """

    if category == "Food":
        cat_col = HICP_class.loc[:, HICP_class.iloc[0, :] == category].columns
    else:
        cat_col = HICP_class.loc[:, HICP_class.iloc[1, :] == category].columns

    print(f"Number of items in {category} group: ", len(cat_col))

    cat_df = HICP_monthly[cat_col]
    if fillna == True:
        cat_df.fillna(0, inplace=True)

    return cat_df


def split_train_test_set(
    X: pd.DataFrame,
    y: pd.DataFrame,
    h: int,
    train_test_split_date=train_test_split_date,
):
    """
    Get the training set for hyperparameter tuning.

    Parameters:
    - X (pd.DataFrame): predictor set
    - y (pd.DataFrame): target
    - h (int): horizon between predictor and target
    - train_test_split_date (optional): date to split training and test set
    Default value is 'train_test_split_date'

    Returns:
    - X_train (pd.DataFrame): The training set features.
    - X_test (pd.DataFrame): The test set features.
    - y_train (pd.DataFrame): The training set target.
    - y_train (pd.DataFrame): The test set target.

    Prints:
    - Current horizon.
    - Period for training set and test set.
    """

    X_train = X[X.index <= train_test_split_date].iloc[:-h, :]
    X_test = X.loc[~X.index.isin(X_train.index)]
    y_train = y[y.index <= train_test_split_date].iloc[h:, :]
    y_test = y[(y.index > train_test_split_date) & (y.index <= max_X_date)]
    print(f"Horizon: {h}")
    print(f"Training predictor period: {X_train.index[0]} to {X_train.index[-1]}")
    print(
        f"Training dependent variable period: {y_train.index[0]} to {y_train.index[-1]}"
    )
    return X_train, X_test, y_train, y_test
