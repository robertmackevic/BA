from pathlib import Path

import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.base import RegressorMixin

from src.data.processing import load_data_from_csv
from src.data.utils import evaluate_model

PRICE = "Electricity_Price"


class NordpoolAutoregForecaster:

    def __init__(self, path_to_csv: Path, forecast_period: int, historical_period: int) -> None:
        self.data: pd.DataFrame = load_data_from_csv(path_to_csv)
        self.forecast_period = forecast_period
        self.historical_period = historical_period
        self.predictions = None
        self.targets = None
        self.test_feature_index = None

    def train(self, regressor: RegressorMixin, test_size: float = .1, verbose: bool = False) -> None:
        price_data = self.data[PRICE]
        price_data_target_count = int(len(price_data) * test_size)
        train_size = len(price_data) - price_data_target_count
        forecast_period_count = price_data_target_count // self.forecast_period
        self.test_feature_index = price_data[train_size:train_size + forecast_period_count * self.forecast_period].index
        y_true, y_pred = [], []

        for i in range(forecast_period_count):
            current_train_data = train_size + i * self.forecast_period
            features = price_data[:current_train_data]
            targets = price_data[current_train_data: current_train_data + self.forecast_period]

            if verbose:
                print(f"Training period: {i + 1}")
                print(f"Train: {features.index.min()} --- {features.index.max()}  (n={len(features)})")
                print(f"Test: {targets.index.min()} --- {targets.index.max()}  (n={len(targets)})")

            forecaster = ForecasterAutoreg(regressor, lags=self.historical_period)
            forecaster.fit(features)
            forecast = forecaster.predict(self.forecast_period)

            y_true.append(targets.to_numpy())
            y_pred.append(forecast)

        print(str(regressor))
        self.targets = np.concatenate(y_true)
        self.predictions = np.concatenate(y_pred)
        evaluate_model(self.targets, self.predictions)
