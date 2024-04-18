### design recursive model for all things:

# like: if == xgb -> do this
# 
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline


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
# 1. Define an objective function to be maximized.
def objective_xgb(trial, X_cat_train, y_cat_train, k_fold = 5):
    """
    For XBG hyperparam tuning

    """
    params = {
        "verbosity": 0,
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'num_boosted_rounds': 100000,
        'early_stopping_rounds': 100,

        'gamma': trial.suggest_float('gamma', 0.0, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.01, 10.0),

        'lambda': trial.suggest_float('lambda', 0.01, 1),
        'alpha': trial.suggest_float('alpha', 0.01, 0.5),

        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    tscv = TimeSeriesSplit(n_splits=k_fold, test_size=24)

    ## Do it explicitly via xgb API:
    cv_mae = [None]*k_fold

    # Add pruning:
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-mae")
        
    for i, (train_index, test_index) in enumerate(tscv.split(X_cat_train, y_cat_train)):
        # split data:
        X_train, X_test = X_cat_train.iloc[train_index], X_cat_train.iloc[test_index]
        y_train, y_test = y_cat_train.iloc[train_index], y_cat_train.iloc[test_index]
        
        # standard scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Transform data into xgboost type:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test)

        # Train model:
        model = xgb.train(params, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback], verbose_eval=False)
        
        y_test_pred = model.predict(dvalid)
        cv_mae[i] = mean_absolute_error(y_test, y_test_pred)
        # saving the individual fold holdout metrics 
        # uncomment this line if you don't want this
        #trial.set_user_attr('split_mae', cv_mae)

    # return the mean of all 5 folds        
    return np.mean(cv_mae)

    ## Do it implicitly via scikit-learn
    # Create a pipeline for preprocessing before XGB:
        
    #model = XGBRegressor(**params) #, callbacks=[pruning_callback]
    #pipeline_ = make_pipeline(StandardScaler(), model)

    # Use cross_val_score to get the mean MAE over multiple splits:
    # here it'll split data as tscv, 
    # then for each fold: it'll scale training data, fit model, then scale validation set, get prediction and put out MAE
    #scores = -cross_val_score(pipeline_, X_cat_train, y_cat_train, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
    #print(scores)
    #return scores.mean()

def hyperparam_tuning_optuna(objective):
    # 1. Define an objective function to be maximized.


    # 2. Create a study object and optimize the objective function.
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    sampler = optuna.samplers.TPESampler(seed=1234)

    study = optuna.create_study(
        pruner=pruner, 
        direction='minimize',
        sampler=sampler
        #storage=storage
    )
    study.optimize(objective, n_trials=1000)
    #run_server(storage, host="localhost", port=8080)
    print("Number of finished trials: {}".format(len(study.trials)))
    

    # Get results:
    results = study.trials_dataframe(attrs=('number','value','duration', 'params'))
    results = results.rename(columns={'value':'mae'})
    results['duration'] = results['duration']/np.timedelta64(1, 's')
    results = results.sort_values(by='mae',ascending=True)
    results.to_csv(f'mae_xgboost_h_{h}.csv',index=False)

    return study.best_params

### Forecast:

# Function for each model:
def xgb_pred(X_train, X_test, y_train, y_test, hyperparam):
    """
    Put it inside the generate_forecast
    """
    # Transform data into xgboost type:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    # Train model:
    model = xgb.train(hyperparam, dtrain, evals=[(dvalid, "validation")], verbose_eval=False)
    
    return model.predict(dvalid)






