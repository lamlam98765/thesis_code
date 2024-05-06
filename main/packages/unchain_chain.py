import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

train_test_split_date = pd.to_datetime("2015-12-31")
max_X_date = pd.to_datetime("2022-12-31")


def transform_back_chained(y_yoy_forecast: pd.Series, y_real: pd.Series) -> pd.Series:
    """
    Transform from HICP y-o-y rate test set back to chain-linked items
    They have to have same indices!!
    """
    last_year = y_real.shift(12)
    last_year = last_year[
        (last_year.index > train_test_split_date) & (last_year.index <= max_X_date)
    ]
    y_chained = (y_yoy_forecast / 100 + 1) * last_year
    return y_chained


def unchain_series(y_chained: pd.Series, y_real: pd.Series):
    """
    Unchain from chain-linked to raw
    """

    y_real = y_real[y_real.index <= "2022-12-31"]
    dec_mask = y_real.index.month == 12
    dec_data = y_real.where(dec_mask, other=np.nan)
    dec_data.ffill(inplace=True)
    dec_data = dec_data[dec_data.index > "2015-12-31"]
    # unchain data:
    y_unchain = y_chained / dec_data
    return y_unchain


def aggregate_to_headline_unchain(food, services, energy, neig, weights: dict):
    """not finish!"""
    total_w = sum(weights.values())
    sum_all = sum(
        [
            food * weights["food"],
            services * weights["services"],
            energy * weights["energy"],
            neig * weights["neig"],
        ]
    )
    return sum_all / total_w
