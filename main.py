from tools.preprocessing import *
from tools.parameters import *

import pandas as pd
import numpy as np

from joblib import dump, load

from sklearn import metrics
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.feature_selection import SelectFromModel, SelectPercentile, SelectKBest, chi2, mutual_info_regression, \
    f_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.decomposition import PCA

from sklearn.ensemble import VotingRegressor, BaggingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.linear_model import SGDRegressor, ElasticNet, PassiveAggressiveRegressor, LassoCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

import xgboost as xgb
import lightgbm as lgb

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


# todo: add gear box bearing efficient from rotor torque and shaft temperature
# todo: remake rotor torque and motor torque with PCA, multivariate outlier for these two features
# todo: explore wind_mill_body_temperature vs generator efficiency
def make_neural_net(input_dim=19, output_dim=1):
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import MeanSquaredError
    model = Sequential()

    model.add(Dense(14, input_dim=input_dim, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
    # model.add(Dropout(0.1))
    model.add(Dense(9, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
    # model.add(Dropout(0.1))
    model.add(Dense(output_dim))  # Final Layer using Softmax

    adam = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=MeanSquaredError())
    model.summary()

    return model


def main():
    # LOAD DATA
    df = read_csv_and_drop_invalid(TRAIN_FILE_NAME, TARGET_FEATURE)

    X, y = df.drop(columns=[TARGET_FEATURE]), df[TARGET_FEATURE]
    # y, X = df.pop(target_feature), df

    # MAKE MODEL PIPELINE
    # clf = lgb.LGBMRegressor(**BEST_PARAMS['lgb'])
    # # 96.7903 score with a standard deviation of 0.40
    # clf = xgb.XGBRegressor()
    # # 96.7685 score with a standard deviation of 0.37
    # clf = RandomForestRegressor(n_estimators=300)
    # # 96.7268 score with a standard deviation of 0.42
    # clf = BaggingRegressor(KNeighborsRegressor())
    # # 91.0921 score with a standard deviation of 0.63
    # clf = SGDRegressor()
    # # 85.3197 score with a standard deviation of 0.80
    # # - 86.0295 score with a standard deviation of 0.67
    # clf = LinearSVR(max_iter=1000, C=1)
    # # 84.2899 score with a standard deviation of 0.84
    # # - 84.8856 score with a standard deviation of 0.80
    # clf = ElasticNet()
    # # 74.3282 score with a standard deviation of 0.67
    # # - 74.4382 score with a standard deviation of 0.78'
    # clf = PassiveAggressiveRegressor()
    # # 75.0200 score with a standard deviation of 3.73
    # clf = LassoCV()
    # # 85.3725 score with a standard deviation of 0.79
    # clf = RidgeCV()
    # # 85.3660 score with a standard deviation of 0.81

    # early_stopping = EarlyStopping(monitor='mean_squared_error', patience=50, verbose=0, restore_best_weights=True)
    # clf = KerasRegressor(make_neural_net, epochs=300, batch_size=500, verbose=1, callbacks=[early_stopping])
    # clf = KerasRegressor(make_neural_net, epochs=300, batch_size=500, verbose=1)

    # clf._estimator_type = 'regressor'
    # nn_pipe = Pipeline([('nn', clf)])

    clf = VotingRegressor(estimators=[
        ('lgb', lgb.LGBMRegressor(**BEST_PARAMS['lgb'])),
        ('xgb', xgb.XGBRegressor(**BEST_PARAMS['xgb'])),
        ('rf', RandomForestRegressor()),
        # ('knn', BaggingRegressor(KNeighborsRegressor()))
        # ('kerasregressor', nn_pipe)
    ])
    # # 96.9950 score with a standard deviation of 0.37

    # clf = StackingRegressor(estimators=[
    #     ('lgb', lgb.LGBMRegressor(**BEST_PARAMS['lgb'])),
    #     ('xgb', xgb.XGBRegressor(**BEST_PARAMS['xgb'])),
    #     ('rf', RandomForestRegressor()),
    #     # ('baggingknn', BaggingRegressor(KNeighborsRegressor())),
    #     # ('lasso', LassoCV()),
    #     # ('ridge', RidgeCV())
    # ], final_estimator=lgb.LGBMRegressor())
    # # 96.5398 score with a standard deviation of 0.04

    model = make_pipeline(preprocessor, clf)

    # FIT, PREDICT AND EVALUATE - SINGULAR
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # score = max(0, 100 * metrics.r2_score(y_test, y_pred))
    # print("%0.4f score" % score)

    # FIT, PREDICT AND EVALUATE - CV
    scores = 100 * cross_val_score(model, X, y, cv=KFold(10, shuffle=True, random_state=SEED), scoring='r2',
                                   verbose=1, n_jobs=-1)
    print("%0.4f score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    # FIT MODEL ON COMPLETE DATA SET
    # model.fit(X, y)
    # model['votingregressor'].named_estimators['kerasregressor'].model.save('keras_model.h5')
    # model['votingregressor'].named_estimators['kerasregressor'].model = None
    # dump(model, 'stacking_regressor.joblib')

    return


if __name__ == '__main__':
    main()
