"""
These functions are designed mostly for hyperparameter tuning step and recursive forecasts and for a specific model
"""

from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import sklearn

# Step 2: Hyperparameter tuning:
# 2.1. Ridge and Lasso, manual fine tuning:


def tuning_gridsearchcv(
    reg: sklearn.base.BaseEstimator,
    grid_space: np.array,
    X_train: pd.DataFrame,
    y_train,
    cv=6,
    test_size=12,
    scoring="neg_root_mean_squared_error",
    scale=True,
):
    """
    Perform hyperparameter tuning using GridSearchCV for Ridge and Lasso regressions.

    Parameters:
    - reg (sklearn.base.BaseEstimator): Regression model to be tuned (e.g., Ridge() or Lasso()).
    - grid_space (np.array): Grid of alpha values to search over for the regression model.
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.DataFrame): The training target values.
    - cv (int, optional): Number of cross-validation splits. The default is 6.
    - test_size (int, optional): Number of test samples in each cross-validation split. The default is 12.
    - scoring (str, optional): Metric to use for evaluation.
    The default is "neg_root_mean_squared_error", meaning negative root mean squared error.
    - scale (bool, optional): if to scale the features before regression. The default is True.

    Returns:
    dict: best hyperparameters found by GridSearchCV.
    """

    # k-fold time series split
    tscv = TimeSeriesSplit(n_splits=cv, test_size=test_size)

    if scale == True:
        pipe = Pipeline(
            [
                ("scaling", StandardScaler()),  # Standardize features:
                ("regression", reg),  # regression
            ]
        )
    else:
        pipe = Pipeline(
            [
                ("regression", reg),
            ]
        )

    param_grid = {"regression__alpha": grid_space}

    grid = GridSearchCV(  # define the Grid search
        pipe, n_jobs=-1, param_grid=param_grid, cv=tscv, scoring=scoring
    )
    grid.fit(X_train, y_train)  # fit to find the best hyperparameter alpha

    return grid.best_params_


def generate_forecast(X, y, N, T, h, hyperparam, model, verbose=0, scale=None):
    """
    Generate recursive forecast
    """
    print(f"Horizon: {h}")
    print("------------------------")
    # standard scale the data, or maybe min max scale it, I'm not sure yet:

    y_pred_series = []
    for i in range(0, T):  # T+1-h
        X_train = X.iloc[: N + i, :]
        y_train = y.iloc[h : N + i + h, :]

        X_test = X.iloc[N + i : N + i + 1, :]
        y_test = y.iloc[N + i + h : N + i + h + 1, :]

        if X_test.index[-1] > X.index[-1] - pd.DateOffset(months=h):
            break
        # Standard scale:
        # For all things

        #### More specific to its own model

        model_here = model(hyperparam)
        model_here.fit(X_train, y_train)
        y_pred = model_here.predict(X_test)

        #####
        if verbose == 1:
            print(
                f"Training period - features: {X_train.index[0]} to {X_train.index[-1]}"
            )
            print(
                f"Training period - target : {y_train.index[0]} to {y_train.index[-1]}"
            )
            print(f"Test period - features: {X_test.index}")
            print(f"Test period - target : {y_test.index}")
            print(f"Forecast: {y_pred}")
            print("-------------------------------------------------------")
        if model == Lasso:
            y_pred_series.append(y_pred[0])
        else:
            y_pred_series.append(y_pred[0][0])

    return y_pred_series


# 1. Define an objective function to be maximized.
def objective_xgb(
    trial, X_cat_train, y_cat_train, k_fold=5, scaler=True, use_all_params=True
):
    """
    For XBG hyperparam tuning

    """
    params = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_boosted_rounds": 10000,
        "early_stopping_rounds": 50,
        "lambda": trial.suggest_float("lambda", 0.01, 1),
        "alpha": trial.suggest_float("alpha", 0.01, 10),
    }

    if use_all_params:
        params.update(
            {
                "gamma": trial.suggest_float("gamma", 0.0, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 10.0),
                # sampling ratio for training data.
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        )
    tscv = TimeSeriesSplit(n_splits=k_fold, test_size=24)

    ## Do it explicitly via xgb API:
    cv_mae = [None] * k_fold

    # Add pruning:
    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, "validation-rmse"
    )

    for i, (train_index, test_index) in enumerate(tscv.split(X_cat_train, y_cat_train)):
        # split data:
        X_train, X_test = X_cat_train.iloc[train_index], X_cat_train.iloc[test_index]
        y_train, y_test = y_cat_train.iloc[train_index], y_cat_train.iloc[test_index]

        # standard scaler:
        if scaler == True:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Transform data into xgboost type:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test)

        # Train model:
        model = xgb.train(
            params,
            dtrain,
            evals=[(dvalid, "validation")],
            callbacks=[pruning_callback],
            verbose_eval=False,
        )

        y_test_pred = model.predict(dvalid)
        cv_mae[i] = mean_absolute_error(y_test, y_test_pred)

    # return the mean of all 5 folds
    return np.mean(cv_mae)


def hyperparam_tuning_optuna(
    objective, X_cat_train, y_cat_train, n_trials=1000, scaler=True, use_all_params=True
):
    # 1. Define an objective function to be maximized.

    # 2. Create a study object and optimize the objective function.
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    sampler = optuna.samplers.TPESampler(seed=1234)

    study = optuna.create_study(
        pruner=pruner,
        direction="minimize",
        sampler=sampler,
        # storage=storage
    )
    study.optimize(
        lambda trial: objective(
            trial,
            X_cat_train,
            y_cat_train,
            k_fold=5,
            scaler=scaler,
            use_all_params=use_all_params,
        ),
        n_trials=n_trials,
    )
    # run_server(storage, host="localhost", port=8080)
    print("Number of finished trials: {}".format(len(study.trials)))
    return study.best_params


### Forecast:


# Function for each model:
def xgb_pred(
    X_train, X_test, y_train, y_test, hyperparam, weight_train=None, weight_test=None
):
    """
    Put it inside the generate_forecast
    """
    # Transform data into xgboost type:
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weight_train)
    dvalid = xgb.DMatrix(X_test, label=y_test, weight=weight_test)

    # Train model:
    model = xgb.train(
        hyperparam, dtrain, evals=[(dvalid, "validation")], verbose_eval=False
    )

    return model.predict(dvalid)


def sarimax_recursive_forecast(y, order, seasonal_order, horizons, N, T):
    """
    Create forecast results after doing SARIMAX recursive forecast, using for baseline models
    Args:
    - y:
    - order: trained order
    - seasonal_order: seasonal order
    - horizon: forecast horizon
    - N: length of training period
    - T: length of test period
    """
    forecast_df = pd.DataFrame()

    # Iterate through the time series data
    for h in horizons:
        forecasts = []
        for i in range(1, T + 1):
            # Define the expanding window training set
            train_data = y[: N + i - h]
            print(f"Horizon {h}, step {i}")

            # Create and fit the SARIMAX model
            model = ARIMA(
                train_data, order=order, seasonal_order=seasonal_order, trend="n"
            )
            model_fit = model.fit()

            # Forecast h step ahead
            pred = model_fit.forecast(steps=h)
            print("Prediction: ", pred)
            forecasts.append(pred.iloc[-1])
        # Assign the forecasted values to a new column in the DataFrame
        forecast_df[f"ar_110_h_{h}"] = forecasts

    # forecast_df['Real_obs'] = head_inf_test.values

    return forecast_df
