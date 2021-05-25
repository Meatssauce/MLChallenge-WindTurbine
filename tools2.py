import re

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

import lightgbm as lgb

# PARAMETERS
SEED = np.random.seed(42)
TARGET_FEATURE = 'windmill_generated_power(kW/h)'
TRAIN_FILE_NAME = 'dataset/train.csv'
TEST_FILE_NAME = 'dataset/test.csv'
MODEL_FILE_NAME = 'final_model.joblib'


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


class WithNamesFuncTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, method):
        self.method = method
        self._feature_names = None

    def fit(self, X, y=None, **fit_params):
        self._feature_names = self.method(self, X).columns.to_list()

        return self

    def transform(self, X, y=None):
        X = self.method(self, X)

        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def get_feature_names(self):
        return self._feature_names


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


def read_csv_and_drop_invalid(file_name, target):
    df = pd.read_csv(file_name)

    # drop duplicates, empty rows and columns and rows with invalid labels
    df = df.dropna(axis=0, how='any', subset=[target])
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df = df.drop_duplicates()

    return df


def my_method(self, X):
    def celsius2kalvin(degrees):
        return degrees + 273.15

    def scaled(arr, about, by):
        return (arr - about) * by + about

    def subset_mean_and_iqr(df, lower_bound, upper_bound):
        subset = df[(df > lower_bound) & (df < upper_bound)]
        return subset.mean(), np.subtract(*np.percentile(subset, [0.75, 0.25]))

    # todo: deal with periodicity in data before removing outliers
    # REMOVE UNITS IN FEATURE NAMES
    new_columns = [re.sub(r'\(.+\)', '', col) for col in X.columns]
    mapper = {X.columns[i]: new_columns[i] for i in range(len(new_columns))}
    X = X.rename(columns=mapper)

    # FEATURE: tracking_id
    X = X.drop(columns=['tracking_id'])

    # FEATURE: datetime
    # convert to days since new year and deal with periodicity
    X['datetime'] = pd.to_datetime(X['datetime'])
    X['days_since_new_year'] = X['datetime'].apply(lambda x: (x - pd.Timestamp(year=x.year-1, month=7, day=1)).days)

    sin_x, cos_x = np.sin(X['days_since_new_year']), np.cos(X['days_since_new_year'])
    X['days_since_new_year_sin'] = sin_x
    X['days_since_new_year_cos'] = cos_x
    distance_vectors = np.array([np.sin(0) - sin_x, np.cos(0) - cos_x]).T
    X['days_since_new_year_normalised_distance'] = np.apply_along_axis(np.linalg.norm, 1, distance_vectors)
    X = X.drop(columns=['datetime'])
    # X = X.drop(columns=['days_since_new_year', 'days_since_new_year_sin', 'days_since_new_year_cos'])

    # FEATURE: wind_speed
    lower_bound, upper_bound = 0, 70
    X['wind_speed'] = X['wind_speed'].where(
        (X['wind_speed'] > lower_bound) & (X['wind_speed'] < upper_bound), np.nan)

    # FEATURE: blades_angle
    # outlier is not outlier. blade angle is probably multimodal
    # two majority groups are perpendicular. 0 degree is 45 degrees to wind direction?
    # todo: look up dealing with multimodal feature in ml
    # X['blades_angle'] = X['blades_angle'].mask(X['blades_angle'] < -10, np.nan)

    # deal with periodicity
    temp = np.deg2rad(X['blades_angle'])
    sin_x, cos_x = np.sin(temp), np.cos(temp)
    X['blades_angle_sin'] = sin_x
    X['blades_angle_cos'] = cos_x
    distance_vectors = np.array([np.sin(0) - sin_x, np.cos(0) - cos_x]).T
    X['blades_angle_normalised_distance'] = np.apply_along_axis(np.linalg.norm, 1, distance_vectors)
    # X = X.drop(columns=['blades_angle', 'blades_angle_sin', 'blades_angle_cos'])

    # FEATURE: motor_torque
    # correct translated and rescaled subset by visual inspection
    lower_bound, wrong_lower_bound = 1000, 500
    scale, translation = 4, 500
    X['motor_torque'] = X['motor_torque'].mask(
        X['motor_torque'] < lower_bound,
        scaled(X['motor_torque'], about=wrong_lower_bound, by=scale) + translation)

    # FEATURE: generator_temperature
    # correct translated and rescaled subset by visual inspection
    lower_bound, wrong_lower_bound = 60, 30
    scale, translation = 2, 30
    X['generator_temperature'] = X['generator_temperature'].mask(
        X['generator_temperature'] < lower_bound,
        scaled(X['generator_temperature'], about=wrong_lower_bound, by=scale) + translation)

    # FEATURE: atmospheric_pressure
    # correct translated and rescaled subset by difference of means

    # compute how much to scale and translate subset
    # subset_mean, subset_iqr = subset_mean_and_iqr(X['atmospheric_pressure'], lower_bound=15000, upper_bound=20000)
    # subset2_mean, subset2_iqr = subset_mean_and_iqr(X['atmospheric_pressure'], lower_bound=100000, upper_bound=150000)
    # scale = subset2_iqr / subset_iqr
    # translation = subset2_mean - subset_mean
    #
    # lower_bound, upper_bound = 15000, 20000
    # X['atmospheric_pressure'] = X['atmospheric_pressure'].mask(
    #     (X['atmospheric_pressure'] > lower_bound) & (X['atmospheric_pressure'] < upper_bound),
    #     scaled(X['atmospheric_pressure'], about=subset_mean, by=scale) + translation)

    # FEATURE: windmill_body_temperature
    X['windmill_body_temperature'] = \
        X['windmill_body_temperature'].mask(X['windmill_body_temperature'] < -73.6, np.nan)

    # FEATURE: wind_direction
    # deal with periodicity
    temp = np.deg2rad(X['wind_direction'])
    sin_x, cos_x = np.sin(temp), np.cos(temp)
    X['wind_direction_sin'] = sin_x
    X['wind_direction_cos'] = cos_x
    distances = np.array([np.sin(0) - sin_x, np.cos(0) - cos_x]).T
    X['wind_direction_norm_distance'] = np.apply_along_axis(np.linalg.norm, 1, distances)
    # X = X.drop(columns=['wind_direction'])
    # X = X.drop(columns=['wind_direction', 'wind_direction_sin', 'wind_direction_cos'])

    # FEATURE: resistance
    # correct translated subset
    lower_bound, translation = 1500, 500
    X['resistance'] = X['resistance'].mask(X['resistance'] < lower_bound, X['resistance'] + translation)
    # remove invalid data post correction (resistance cannot be negative)
    X['resistance'] = X['resistance'].mask(X['resistance'] <= 0, np.nan)

    # FEATURE: rotor_torque
    # correct translated and rescaled subset by visual inspection
    lower_bound, wrong_lower_bound = 20, 5
    scale, translation = 2, 15
    X['rotor_torque'] = X['rotor_torque'].mask(
        X['rotor_torque'] < lower_bound,
        scaled(X['rotor_torque'], about=wrong_lower_bound, by=scale) + translation)

    # FEATURE: blade_length
    X['blade_length'] = X['blade_length'].mask(X['blade_length'] <= 0, np.nan)

    # FEATURE: windmill_height
    X['windmill_height'] = X['windmill_height'].mask(X['windmill_height'] <= 0, np.nan)

    # FEATURE: blade_area
    X['blade_area'] = X['blade_length'] * X['blade_breadth']

    # # FEATURE: swept_area
    # X['swept_area'] = np.pi * X['blade_length'].pow(2)

    # # FEATURE: excess_heat
    # # find sum of differences of component temperatures from environment temperature (mean of area and atmospheric)
    # X['excess_heat'] = X['shaft_temperature'] + X['gearbox_temperature'] + X['engine_temperature'] + \
    #     X['generator_temperature'] - 4 * (X['atmospheric_temperature'] + X['area_temperature']) / 2

    # FEATURE: air_density
    p = X['atmospheric_pressure']                       # absolute pressure, Pascal
    R = 287                                             # specific gas constant of air, J/(kg*K)
    T = celsius2kalvin(X['atmospheric_temperature'])    # atmospheric temperature, K
    X['air_density'] = p / (T * R)

    # FEATURE: air_mass_flow_rate
    X['air_mass_flow_rate'] = X['air_density'] * X['wind_speed']

    # X = X.drop(columns=['air_density', 'wind_speed'])

    # UNDO REMOVING UNITS IN FEATURE NAMES
    mapper = {v: k for k, v in mapper.items()}
    X = X.rename(columns=mapper)

    return X


columns_without_outliers = ['days_since_new_year', 'days_since_new_year_sin', 'wind_direction_cos',
                            'blades_angle(°)', 'blades_angle_sin', 'blades_angle_cos', 'blades_angle_norm',
                            'blade_area',
                            'wind_direction(°)', 'wind_direction_sin', 'wind_direction_cos', 'wind_direction_norm']

preprocessor = make_pipeline(
    WithNamesFuncTransformer(my_method),
    ColumnTransformer(transformers=[
        ('numerical', make_pipeline(
            # ColumnTransformer(transformers=[
            #     ('norm', make_pipeline(SimpleImputer(strategy="median"), OutlierNullifier()),
            #      [x for x in numerical_columns if x not in outlier_skipped]),
            #     ('exception', make_pipeline(SimpleImputer(strategy="median")),
            #      outlier_skipped)
            # ]),
            OutlierNullifier(),
            SimpleImputer(strategy="median"),
            StandardScaler()
        ), selector(dtype_include=np.number)),
        ('categorical', make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder()
        ), selector(dtype_include=['category', object, 'bool'])),
    ]),
    # PCA(25),
    SelectFromModel(lgb.LGBMRegressor()),
    # SelectKBest(f_regression, k=25)
)
