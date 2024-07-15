# Master's thesis: Forecasting German Inflation by item-level price data

## Introduction:
I forecast HICP headline inflation using over 600 disaggregated CPI indices, using models designed for high-dimensional cases, including: dimensional reduction techniques - PCR (Principal Component Regression), shrinkage models - Ridge and Lasso, XGBoost.

## Folder structure:
- `data`: includes all of the data and results.

`forecast_results`: contain forecasts for four categories.

`headline_forecast`: contain forecasts for headline inflation.

`hicp_cat_raw`: raw HICP data for four categoies and their weights.

`preprocessed`: year-on-year transformed HICP rate.

`report_rmse`: RMSE for comparison.

- `main`: all the notebooks and packages

`packages`: my own packages.

Notebooks: run the models

- `model`: save xgboost plain (notebook 5.1) hyperparameters.

- `requirement.txt`: all the necessary package versions.



