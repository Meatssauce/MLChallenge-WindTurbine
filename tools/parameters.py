import numpy as np

# PARAMETERS
SEED = np.random.seed(42)
TARGET_FEATURE = 'windmill_generated_power(kW/h)'
TRAIN_FILE_NAME = 'dataset/train.csv'
TEST_FILE_NAME = 'dataset/test.csv'
MODEL_FILE_NAME = 'final_model.joblib'

LGB_SELECTOR_PARAMS = {'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 300, 'num_leaves': 31}
BEST_PARAMS = {'lgb': {'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 300, 'num_leaves': 31},
               'xgb': {'learning_rate': 0.1, 'max_depth': 9},
               'rf': {},
               'knn': {'leaf_size': 30, 'n_neighbors': 8}
               }
