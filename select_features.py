from tools.preprocessing import *
from boruta import BorutaPy

best_params = {
    'num_leaves': 76,
    'max_depth': 9,
    'learning_rate': 0.06465101257294228,
    'n_estimators': 213,
    'min_child_weight': 0.007299462676584353,
    'min_child_samples': 24,
    'subsample': 0.6658929378716963,
    'subsample_for_bin': 266280,
    'subsample_freq': 20,
    'colsample_bytree': 0.7440805406257257,
    'reg_alpha': 8.955681524535454,
    'reg_lambda': 3.712127820375468}
estimator = lgb.LGBMRegressor(random_state=42, **best_params)
step1 = {'Relevant Features': {'cv': 5,
                               'estimator': estimator,
                               'n_estimators': 1000,
                               'max_iter': 50,
                               'verbose': 0,
                               'random_state': 42}}
steps = [step1]


selector = BorutaPy(**{'estimator': lgb.LGBMRegressor(**best_params),
                       'n_estimators': 1000,
                       'max_iter': 100,
                       'verbose': 50,
                       'random_state': 42})
