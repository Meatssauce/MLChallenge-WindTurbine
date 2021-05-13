from tools import *
from joblib import dump, load

# LOAD DATA
X_pred = pd.read_csv(TEST_FILE_NAME)
model = load(MODEL_FILE_NAME)

# PREDICT
y_pred = model.predict(X_pred)

# FORMAT AND SAVE SUBMISSION
submission = X_pred[['tracking_id', 'datetime']]
submission[TARGET_FEATURE] = y_pred
submission.to_csv('dataset/submission.csv', index=False)
