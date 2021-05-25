from tools.preprocessing import *
from tools.parameters import *

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, KFold

import xgboost as xgb


def main():
    # Make a composite estimator that includes preprocessing
    # todo: see if make_union names are sensible
    from sklearn.compose import ColumnTransformer, make_column_selector as selector
    numerical_selector = selector(dtype_include=np.number)
    categorical_selector = selector(dtype_include=['category', object, 'bool'])
    # pipe = make_pipeline(
    #     Preprocessor(my_method),
    #     ColumnTransformer(transformers=[
    #         ('numerical', make_pipeline(
    #             SimpleImputer(strategy="median"),
    #             OutlierNullifier(),
    #             StandardScaler()
    #         ), numerical_selector),
    #         ('categorical', make_pipeline(
    #             SimpleImputer(strategy='most_frequent'),
    #             OrdinalEncoder()
    #         ), categorical_selector),
    #     ]),
    #     # xgb.XGBRegressor(objective='reg:squarederror', random_state=seed, verbosity=0, scoring='r2', n_jobs=-1)
    # )

    df = read_csv_and_drop_invalid(TRAIN_FILE_NAME, TARGET_FEATURE)

    X, y = df.drop(columns=[TARGET_FEATURE]), df[TARGET_FEATURE]
    # y, X = df.pop(target_feature), df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

    prp = WithNamesFuncTransformer(my_method)
    numerical_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    outlier_remover = OutlierNullifier()
    scaler = StandardScaler()
    encoder = OrdinalEncoder()

    X = prp.fit_transform(X)
    numerical_features = X.select_dtypes(include=[np.number])
    # columns_with_outliers = [col for col in numerical_features if col not in columns_without_outliers]
    # X[columns_with_outliers] = pd.DataFrame(outlier_remover.fit_transform(
    #         X[columns_with_outliers]), columns=columns_with_outliers, index=numerical_features.index)
    X[numerical_features.columns] = pd.DataFrame(outlier_remover.fit_transform(
        X[numerical_features.columns]), columns=numerical_features.columns, index=numerical_features.index)
    X[numerical_features.columns] = pd.DataFrame(numerical_imputer.fit_transform(
        X[numerical_features.columns]), columns=numerical_features.columns, index=numerical_features.index)
    X[numerical_features.columns] = pd.DataFrame(scaler.fit_transform(
            X[numerical_features.columns]), columns=numerical_features.columns, index=numerical_features.index)
    categorical_features = X.select_dtypes(include=[object, 'bool', 'category'])
    X[categorical_features.columns] = pd.DataFrame(categorical_imputer.fit_transform(
        X[categorical_features.columns]), columns=categorical_features.columns, index=categorical_features.index)
    X[categorical_features.columns] = pd.DataFrame(encoder.fit_transform(
        X[categorical_features.columns]), columns=categorical_features.columns, index=categorical_features.index)
    generate_report(X, y, TARGET_FEATURE)

    X_test = prp.transform(X_test)
    numerical_features = X_test.select_dtypes(include=[np.number])
    columns_with_outliers = [col for col in numerical_features if col not in columns_without_outliers]
    X_test[columns_with_outliers] = pd.DataFrame(outlier_remover.transform(
        X_test[columns_with_outliers]), columns=columns_with_outliers, index=numerical_features.index)
    X_test[numerical_features.columns] = pd.DataFrame(numerical_imputer.transform(
        X_test[numerical_features.columns]), columns=numerical_features.columns, index=numerical_features.index)
    X_test[numerical_features.columns] = pd.DataFrame(scaler.transform(
            X_test[numerical_features.columns]), columns=numerical_features.columns, index=numerical_features.index)
    categorical_features = X_test.select_dtypes(include=[object, 'bool', 'category'])
    X_test[categorical_features.columns] = pd.DataFrame(categorical_imputer.transform(
        X_test[categorical_features.columns]), columns=categorical_features.columns, index=categorical_features.index)
    X_test[categorical_features.columns] = pd.DataFrame(encoder.transform(
        X_test[categorical_features.columns]), columns=categorical_features.columns, index=categorical_features.index)

    # todo: find out why code below doesn't work after line 180
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=SEED, verbosity=0, scoring='r2', n_jobs=-1)
    model.fit(X_train, y_train)
    from sklearn import metrics
    y_pred = model.predict(X_test)
    score = max(0, 100 * metrics.r2_score(y_test, y_pred))
    print("%0.4f score" % score)

    # TRAIN PREDICT EVALUATE - SINGULAR
    # pipe.fit(X_train, y_train)
    # from sklearn import metrics
    # y_pred = pipe.predict(X_test)
    # score = max(0, 100 * metrics.r2_score(y_test, y_pred))
    # print("%0.4f score" % score)

    # TRAIN PREDICT EVALUATE - CV
    # from sklearn.model_selection import cross_val_score
    # scores = 100 * cross_val_score(pipe, X, y, cv=10, scoring='r2', verbose=1, n_jobs=-1)
    # print("%0.4f score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    return


if __name__ == '__main__':
    main()
