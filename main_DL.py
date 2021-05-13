import re

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

import xgboost as xgb
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2


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
                self.quantiles[i] = np.quantile(X[i], 0.25), np.quantile(X[i], 0.75)
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
                X[column] = X[column].mask((X[column] < q1 - 1.5 * iqr) | (X[column] > q3 + 1.5 * iqr), np.nan)

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
    X['days_since_new_year'] = X['datetime'].apply(lambda x: (x - pd.Timestamp(year=x.year, month=1, day=1)).days)
    X = X.drop(columns=['datetime'])

    # FEATURE: blades_angle
    # outlier is not outlier. blade angle is probably multimodal
    # two majority groups are perpendicular. 0 degree is 45 degrees to wind direction?
    # todo: look up dealing with multimodal feature in ml

    # FEATURE: windmill_body_temperature
    X['windmill_body_temperature'] = \
        X['windmill_body_temperature'].mask(X['windmill_body_temperature'] < -73.6, np.nan)

    # FEATURE: resistance
    X['resistance'] = X['resistance'].mask(X['resistance'] <= 0, np.nan)

    # FEATURE: blade_length
    X['blade_length'] = X['blade_length'].mask(X['blade_length'] <= 0, np.nan)

    # FEATURE: windmill_height
    X['windmill_height'] = X['windmill_height'].mask(X['windmill_height'] <= 0, np.nan)

    # UNDO REMOVING UNITS IN FEATURE NAMES
    mapper = {v: k for k, v in mapper.items()}
    X = X.rename(columns=mapper)

    return X


# todo: see if pipeline replaced with neural net performs better
def make_neural_net(input_dim, output_dim=1):
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import MeanSquaredError
    model = Sequential()

    model.add(Dense(18, input_dim=input_dim, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
    model.add(Dropout(0.1))
    model.add(Dense(9, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)))
    model.add(Dropout(0.1))
    model.add(Dense(output_dim))  # Final Layer using Softmax

    adam = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=adam, metrics=MeanSquaredError())
    model.summary()

    return model


def main():
    # Make a composite estimator that includes preprocessing
    # todo: see if make_union names are sensible
    # from mlxtend.feature_selection import ColumnSelector
    from sklearn.compose import ColumnTransformer, make_column_selector as selector
    # numerical_selector = selector(dtype_include=np.number)
    # categorical_selector = selector(dtype_include=['category', object, 'bool'])
    model = make_neural_net(35)
    pipe = make_pipeline(
        Preprocessor(my_method),
        ColumnTransformer(transformers=[
            ('numerical', make_pipeline(
                SimpleImputer(strategy="median"),
                # ColumnTransformer([
                #     ('non-speical', OutlierNullifier(), ColumnSelector(['blades_angle(°)']))
                # ]),
                # OutlierNullifier(),
                StandardScaler()
            ), selector(dtype_include=np.number)),
            ('categorical', make_pipeline(
                SimpleImputer(strategy='most_frequent'),
                OneHotEncoder()
            ), selector(dtype_include=['category', object, 'bool'])),
        ]),
    )

    df = pd.read_csv('dataset/train.csv')

    # drop duplicates, empty rows and columns and rows with invalid labels
    df = df.dropna(axis=0, how='any', subset=[target_feature])
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df = df.drop_duplicates()

    X, y = df.drop(columns=[target_feature]), df[target_feature].copy()
    # y, X = df.pop(target_feature), df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    # TRAIN PREDICT EVALUATE - SINGULAR
    from tensorflow.keras.callbacks import EarlyStopping
    X_train = pipe.fit_transform(X_train, y_train)
    X_test = pipe.transform(X_test)
    early_stopping = EarlyStopping(monitor='val_mean_squared_error', patience=50, verbose=1, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, callbacks=[early_stopping])

    from sklearn import metrics
    y_pred = model.predict(X_test)
    score = max(0, 100 * metrics.r2_score(y_test, y_pred))
    print("%0.4f score" % score)

    # TRAIN PREDICT EVALUATE - CV
    # from sklearn.model_selection import cross_val_score
    # scores = 100 * cross_val_score(pipe, X, y, cv=10, scoring='r2', verbose=1, n_jobs=-1)
    # print("%0.4f score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    return


if __name__ == '__main__':
    main()
