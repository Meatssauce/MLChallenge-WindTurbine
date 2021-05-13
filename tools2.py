import re

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.neighbors import LocalOutlierFactor


class OutlierNullifier(TransformerMixin):
    def __init__(self, **kwargs):
        """
        Create a transformer to remove outliers.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """

        self.quantiles = {}

    def fit(self, X, y=None, **fit_params):
        if isinstance(X, np.ndarray):
            for i in range(X.shape[1]):
                self.quantiles[i] = np.quantile(X[i], [0.25, 0.75])
        else:
            for column in X.columns:
                self.quantiles[column] = X[column].quantile(0.25), X[column].quantile(0.75)

        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            for i in range(X.shape[1]):
                q1, q3 = self.quantiles[i]
                iqr = q3 - q1
                X[i] = np.where((X[i] < q1 - 1.5 * iqr) | (X[i] > q3 + 1.5 * iqr), np.nan, X[i])
        else:
            for column in X.columns:
                q1, q3 = self.quantiles[column]
                iqr = q3 - q1
                X[column] = np.where((X[column] < q1 - 1.5 * iqr) | (X[column] > q3 + 1.5 * iqr), np.nan, X[column])

        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)


class Preprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, method):
        self.method = method

    def fit(self, X, y=None, **fit_params):
        self.method(self, X)

        return self

    def transform(self, X, y=None):
        X = self.method(self, X)

        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, *dtypes):
        self.dtypes = list(dtypes)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=self.dtypes)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


def generate_report(X, y, target_feature):
    from matplotlib import pyplot as plt
    from dataprep.eda import create_report
    df = X
    df[target_feature] = y
    create_report(df).show_browser()
    return


def my_method(self, X):
    # todo: deal with periodicity in data before removing outliers
    # REMOVE UNITS IN FEATURE NAMES
    new_columns = []
    for column in X.columns:
        new_columns.append(re.sub(r'\(.+\)', '', column))
    mapper = {X.columns[i]: new_columns[i] for i in range(len(new_columns))}
    X = X.rename(columns=mapper)

    # FEATURE: tracking_id
    X = X.drop(columns=['tracking_id'])

    # FEATURE: datetime
    X['datetime'] = pd.to_datetime(X['datetime'])
    X['days_since_mid_year'] = X['datetime'].apply(lambda x: (x - pd.Timestamp(year=x.year - 1, month=7, day=1)).days)
    sin_x, cos_x = np.sin(X['days_since_mid_year']), np.cos(X['days_since_mid_year'])
    X['days_since_mid_year_sin'] = sin_x
    X['days_since_mid_year_cos'] = cos_x
    distances = np.array([np.sin(0) - sin_x, np.cos(0) - cos_x]).T
    X['days_since_mid_year_norm_distance'] = np.apply_along_axis(np.linalg.norm, 1, distances)
    X = X.drop(columns=['datetime'])

    # FEATURE: wind_speed
    # X['wind_speed'] = X['wind_speed'].mask((X['wind_speed'] > 80) & (X['wind_speed'] < 110), np.nan)

    # FEATURE: blades_angle
    # outlier is not outlier. blade angle is probably multimodal
    # two majority groups are perpendicular. 0 degree is 45 degrees to wind direction?
    # todo: look up dealing with multimodal feature in ml
    # X['blades_angle'] = X['blades_angle'].mask(X['blades_angle'] < -10, np.nan)

    temp = np.deg2rad(X['blades_angle'])
    sin_x, cos_x = np.sin(temp), np.cos(temp)
    X['blades_angle_sin'] = sin_x
    X['blades_angle_cos'] = cos_x
    distances = np.array([np.sin(0) - sin_x, np.cos(0) - cos_x]).T
    X['blades_angle_norm_distance'] = np.apply_along_axis(np.linalg.norm, 1, distances)
    # X = X.drop(columns=['blades_angle'])

    # FEATURE: motor_torque
    # repartition
    X['motor_torque'] = X['motor_torque'].mask(X['motor_torque'] < 1000, (X['motor_torque'] - 500) * 4 + 1000)

    # FEATURE: generator_temperature
    # repartition
    X['generator_temperature'] = X['generator_temperature'].mask(
        X['generator_temperature'] < 60, (X['generator_temperature'] - 30) * 2 + 60)

    # FEATURE: atmospheric_pressure
    # X['atmospheric_pressure'] = X['atmospheric_pressure'].mask(X['atmospheric_pressure'] < 100000, np.nan)

    # FEATURE: windmill_body_temperature
    X['windmill_body_temperature'] = \
        X['windmill_body_temperature'].mask(X['windmill_body_temperature'] < -73.6, np.nan)

    # FEATURE: wind_direction
    temp = np.deg2rad(X['wind_direction'])
    sin_x, cos_x = np.sin(temp), np.cos(temp)
    X['wind_direction_sin'] = sin_x
    X['wind_direction_cos'] = cos_x
    distances = np.array([np.sin(0) - sin_x, np.cos(0) - cos_x]).T
    X['wind_direction_norm_distance'] = np.apply_along_axis(np.linalg.norm, 1, distances)
    # X = X.drop(columns=['wind_direction'])

    # FEATURE: resistance
    # repartition
    X['resistance'] = X['resistance'].mask(X['resistance'] < 1500, X['resistance'] + 500)
    # cannot be negative
    X['resistance'] = X['resistance'].mask(X['resistance'] <= 0, np.nan)

    # FEATURE: rotor_torque
    # repartition
    X['rotor_torque'] = X['rotor_torque'].mask(X['rotor_torque'] < 20, (X['rotor_torque'] - 5) * 2 + 20)

    # FEATURE: blade_length
    X['blade_length'] = X['blade_length'].mask(X['blade_length'] <= 0, np.nan)

    # FEATURE: windmill_height
    X['windmill_height'] = X['windmill_height'].mask(X['windmill_height'] <= 0, np.nan)

    # FEATURE: cross_sectional_area
    X['blade_area'] = X['blade_length'] * X['blade_breadth']

    # # FEATURE: swept_area
    # X['swept_area'] = np.pi * X['blade_length'].pow(2)

    # UNDO REMOVING UNITS IN FEATURE NAMES
    mapper = {v: k for k, v in mapper.items()}
    X = X.rename(columns=mapper)

    return X


columns_without_outliers = ['days_since_mid_year', 'days_since_mid_year_sin', 'wind_direction_cos',
                            'blades_angle(°)', 'blades_angle_sin', 'blades_angle_cos', 'blades_angle_norm',
                            'blade_area',
                            'wind_direction(°)', 'wind_direction_sin', 'wind_direction_cos', 'wind_direction_norm']
