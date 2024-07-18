# Master's thesis: Forecasting German Inflation by item-level price data

## Introduction:
I forecast HICP German headline inflation using over 600 disaggregated CPI indices, using models designed for high-dimensional cases, including: dimensional reduction techniques - PCR (Principal Component Regression), shrinkage models - Ridge and Lasso, XGBoost, experimenting with different input and preprocessing steps.

My paper: [https://drive.google.com/file/d/1M_ixf295cfQncFfIYWje2R5ve2huVdBg/view?usp=share_link](url)

## Folder structure:
- `data`: includes all of the data and results.

`forecast_results`: contain forecasts for four categories.

`headline_forecast`: contain forecasts for headline inflation.

`hicp_cat_raw`: raw HICP data for four categoies and their weights.

`preprocessed`: year-on-year transformed HICP rate.

`report_rmse`: RMSE for comparison.

- `main`: all the notebooks and packages

`packages`: my own packages.

Notebooks: run the models (I choose notebooks because they are easier for reviewer to check the validity.)

`1.1.AR_model_headline.ipynb` and `1.2.AR_1_categories.ipynb` are for benchmark models.

`2_Preprocess_HICP_all.ipynb` preprocesses data and do some Exloratory Data Analysis and visualization.

Notebooks from `3.` to `6.` are my experiments. 

Notebooks `7.Compare_models_categories.ipynb` and `8.8.Aggregation_compare_headline.ipynb` are for evaluation.

- `model`: save xgboost plain (notebook 5.1) hyperparameters.

- `requirement.txt`: all the necessary package versions.



