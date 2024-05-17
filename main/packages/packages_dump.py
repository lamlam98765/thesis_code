

class RecursiveForecast:
    """
    Create forcast for 1 horizon
    """

    def __init__(
        self, X: pd.DataFrame, y: pd.Series, h: int, model, hyperparameter
    ) -> None:
        self.X = X
        self.y = y
        self.split_date = train_test_split_date
        self.h = h
        self.model = model
        self.hyperparameter = hyperparameter

        print(
            f"Train test split date: {self.split_date}, model {self.model}, hyperparameter: {self.hyperparameter}"
        )

    def train_test_split(self):
        """
        Split the train and test set based on the predetermined date

        """
        X_train = self.X[self.X.index <= self.split_date].iloc[
            : -self.h, :
        ]  # as yt = f(Xt-1)
        X_test = self.X[~self.X.index.isin(X_train.index)]

        y_train = self.y[self.y.index <= self.split_date][self.h :]
        y_test = self.y[self.y.index > self.split_date]
        N, T = len(X_train), len(X_test)

        print(f"Horizon: {self.h}")
        print(f"Training predictor period: {X_train.index[0]} to {X_train.index[-1]}")
        print(
            f"Training dependent variable period: {y_train.index[0]} to {y_train.index[-1]}"
        )
        print(f"Test predictor period: {X_test.index[0]} to {X_test.index[-1]}")
        print(
            f"Test dependent variable period: {y_test.index[0]} to {y_test.index[-1]}"
        )
        print("--------------------------------------------")

        return N, T, #y_test

    def generate_forecast(self, N: int, T: int) -> pd.DataFrame:
        y_pred_series = []
        for i in range(1, T):
            X_train = self.X.iloc[: N+i, :]
            y_train = self.y.iloc[self.h : N+self.h+i, :]

            X_test = self.X.iloc[N+i:N+i+1, :]

            # forecast horizon h:
            print(
                f"Predictor training period: {X_train.index[0]} to {X_train.index[-1]}"
            )
            print(f"Forecast target period: {y_train.index[0]} to {y_train.index[-1]}")
            print(f"Predictor test period: {X_test.index}")

            model = self.model(self.hyperparameter)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"Forecast: ")
            print(y_pred)
            # if len(y_pred) < 3:
            # Pad with zeros to make it a length of 3
            #    y_pred = np.pad(y_pred, (0, 3 - len(y_pred)), 'constant')
            # forecast_df.iloc[i-1, :] = y_pred
            print("-------------------------------------------------------")
            y_pred_series.append(y_pred[0, 0])
        return y_pred_series

    def concat_forecast(self):
        """
        Put all functions above into 1 pipeline
        """
        N, T = self.train_test_split()
        # init_forecast_df = self.create_forecast_df(N = N)
        y_pred_series = self.generate_forecast(N=N, T=T)
        # final_forecast = self.chop_forecast_to_fit(forecast_df, y_test=y_test)
        return y_pred_series



def concat_all_horizons(pred_1, pred_2, pred_3, y_test, model, max_horizon=3):
    """
    Concatinate all horizons into 1 df
    """
    forecast_result = pd.DataFrame(
        columns=[f"{model}_h_{i}" for i in range(1, max_horizon + 1)],
        index=y_test[y_test.index < "2023-01-31"].index,
    )

    forecast_result.iloc[:, 0] = pred_1
    forecast_result.iloc[:, 1] = pred_2
    forecast_result.iloc[:, 2] = pred_3
    return forecast_result