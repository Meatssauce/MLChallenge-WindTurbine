import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from tools.parameters import *
from tools.preprocessing import *


def tune_knn():
    # Load data
    df = read_csv_and_drop_invalid(TRAIN_FILE_NAME, TARGET_FEATURE)

    X, y = df.drop(columns=[TARGET_FEATURE]), df[TARGET_FEATURE]

    # Initialise estimator
    estimator = Pipeline([('preprocessor', preprocessor),
                          ('estimator', KNeighborsRegressor())])

    # Define parameter grid
    params = {'estimator__n_neighbors': [5, 8, 10],
              # 'estimator__weights': ['uniform', 'distance'],
              'estimator__leaf_size': [30, 50, 70]
              }

    estimator.set_params(**params)

    # Initialize GridSearch object
    gscv = GridSearchCV(estimator, params, cv=KFold(10, shuffle=True, random_state=SEED), scoring='r2',
                        verbose=1, n_jobs=-1)
    gscv.fit(X, y)
    best_params = gscv.best_params_
    best_score = gscv.best_score_

    print("%0.4f score" % best_score)

    return best_params


def tune_rf():
    # Load data
    df = read_csv_and_drop_invalid(TRAIN_FILE_NAME, TARGET_FEATURE)

    X, y = df.drop(columns=[TARGET_FEATURE]), df[TARGET_FEATURE]

    # Initialise estimator
    estimator = make_pipeline(preprocessor, RandomForestRegressor())

    # Define parameter grid
    params = {'randomforestregressor__max_depth': [None, 3, 6, 9],
              'randomforestregressor__criterion': ['mse', 'mae'],
              'randomforestregressor__n_estimators': [100, 200, 300]
              }

    estimator.set_params(**params)

    # Initialize GridSearch object
    gscv = GridSearchCV(estimator, params, cv=KFold(10, shuffle=True, random_state=SEED), scoring='r2',
                        verbose=1, n_jobs=-1)
    gscv.fit(X, y)
    best_params = gscv.best_params_
    best_score = gscv.best_score_

    print("%0.4f score" % best_score)

    return best_params


def tune_xgb():
    # Load data
    df = read_csv_and_drop_invalid(TRAIN_FILE_NAME, TARGET_FEATURE)

    X, y = df.drop(columns=[TARGET_FEATURE]), df[TARGET_FEATURE]

    # Initialise estimator
    estimator = make_pipeline(preprocessor, xgboost.XGBRegressor())

    # Define parameter grid
    params = {'xgbregressor__max_depth': [3, 6, 9],
              'xgbregressor__learning_rate': [0.1, 0.3, 0.6, 1],
              # 'xgbregressor__n_estimators': [100, 200, 300]
              }

    estimator.set_params(**params)

    # Initialize GridSearch object
    gscv = GridSearchCV(estimator, params, cv=KFold(10, shuffle=True, random_state=SEED), scoring='r2',
                        verbose=1, n_jobs=-1)
    gscv.fit(X, y)
    best_params = gscv.best_params_
    best_score = gscv.best_score_

    print("%0.4f score" % best_score)

    return best_params


def tune_lgb():
    # Load data
    df = read_csv_and_drop_invalid(TRAIN_FILE_NAME, TARGET_FEATURE)

    X, y = df.drop(columns=[TARGET_FEATURE]), df[TARGET_FEATURE]

    # Initialise estimator
    estimator = make_pipeline(preprocessor, lgb.LGBMRegressor())

    # Define parameter grid
    params = {'lgbmregressor__num_leaves': [31],
              'lgbmregressor__max_depth': [-1],
              'lgbmregressor__learning_rate': [0.01, 0.1, 1],
              'lgbmregressor__n_estimators': [100, 200, 300],
              # 'lgbmregressor__min_child_weight': config.lgbmregressor__min_child_weight,
              # 'lgbmregressor__min_child_samples': config.lgbmregressor__min_child_samples,
              # 'lgbmregressor__subsample': config.lgbmregressor__subsample,
              # 'lgbmregressor__subsample_freq': config.lgbmregressor__subsample_freq,
              # 'lgbmregressor__reg_alpha': config.lgbmregressor__reg_alpha,
              # 'lgbmregressor__reg_lambda': config.lgbmregressor__reg_lambda
              }

    estimator.set_params(**params)

    # Initialize GridSearch object
    gscv = GridSearchCV(estimator, params, cv=KFold(10, shuffle=True, random_state=SEED), scoring='r2',
                        verbose=1, n_jobs=-1)
    gscv.fit(X, y)
    best_params = gscv.best_params_
    best_score = gscv.best_score_

    print("%0.4f score" % best_score)

    return best_params


if __name__ == '__main__':
    best_params = {'lgb': tune_lgb(),
                   'xgb': tune_xgb(),
                   'knn': tune_knn()}
    print(best_params)
    # print(tune_rf())
