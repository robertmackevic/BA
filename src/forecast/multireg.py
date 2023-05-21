import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.processing import load_data_from_csv
from src.data.utils import evaluate_model

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

TARGET = "Electricity_Price"
FACTORS = ["Consumption", "Production", "Exchange", "Gasoline_Price"]


class NordpoolMultiOutputForecaster:
    def __init__(self, path_to_csv: Path, forecast_period: int, historical_period: int) -> None:
        self.data: pd.DataFrame = load_data_from_csv(path_to_csv)
        self.dataset = self.data.copy()
        self.forecast_period = forecast_period
        self.historical_period = historical_period
        self.data_is_prepared = False
        self.train_features = None
        self.train_targets = None
        self.test_targets = None
        self.test_features = None
        self.target_features = None
        self.predictions = None
        self.targets = None
        self.test_feature_index = None

    def prepare_multi_output_data(self, test_size: float = .1) -> None:
        self.dataset = self.dataset.drop(columns="Wind_Power")
        self.add_history_for_columns(column_names=[TARGET], period=self.historical_period, step=1)
        self.add_history_for_columns(column_names=FACTORS, period=self.historical_period, step=1)
        self.dataset = self.dataset.drop(columns=FACTORS)
        self.remove_trailling_and_leading_nans()
        self.create_test_train_datasets(TARGET, target_period=self.forecast_period, test_size=test_size)
        self.data_is_prepared = True

    def add_history_for_columns(self, column_names: List[str], period: int, step: int) -> None:
        for name in column_names:
            for i in range(1, period + 1, step):
                self.dataset[f"{name}_{i}H_AGO"] = self.dataset[name].shift(i)

    def remove_trailling_and_leading_nans(self) -> None:
        for column in self.dataset.columns:
            column_series = self.dataset[column]
            trailling = column_series.first_valid_index()
            leading = column_series.last_valid_index()
            self.dataset = self.dataset[trailling:leading]

    def create_test_train_datasets(self, target_column: str, target_period: int, test_size: float) -> None:
        def create_target_arrays(target_column: str, target_period: int):
            targets = self.dataset[target_column].values
            if target_period < 2:
                return targets
            return np.array([targets[i:i + target_period] for i in range(len(self.dataset) - target_period)])

        test_size = int(len(self.dataset) * test_size)
        train_size = len(self.dataset) - test_size
        features = self.dataset.drop(columns=[target_column]).values[:-target_period]
        targets = create_target_arrays(target_column, target_period)
        index_elements = self.dataset.index[train_size: -target_period]
        elements_to_remove = len(index_elements) % 24
        self.test_feature_index = index_elements[:-elements_to_remove]

        self.train_features = features[:train_size]
        self.test_features = features[train_size:] if target_period < 2 else features[train_size:-target_period]
        self.train_targets = targets[:train_size]
        self.test_targets = targets[train_size:-target_period]

        summary = pd.DataFrame({
            "Type": ["Train", "Test"],
            "Features": [self.train_features.shape, self.test_features.shape],
            "Targets": [self.train_targets.shape, self.test_targets.shape]
        })

        print(summary)
        print("\n")

    def train(self, regressor: RegressorMixin, test_size: float = .1) -> None:
        if not self.data_is_prepared:
            self.prepare_multi_output_data(test_size)

        pipeline = make_pipeline(
            SimpleImputer(missing_values=0),
            StandardScaler(),
            regressor
        )

        pipeline.fit(self.train_features, self.train_targets)
        forecast = pipeline.predict(self.test_features)

        y_pred = forecast[::self.forecast_period].flatten()
        y_true = self.test_targets[::self.forecast_period].flatten()
        self.targets = y_true
        self.predictions = y_pred
        print(str(regressor))
        evaluate_model(y_true, y_pred)
