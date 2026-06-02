import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from experiments.logger import log_experiment

TRAIN_FILE = "train_features_v2.csv"
VALIDATION_FILE = "val_features_v2.csv"
FEATURE_SET_VERSION = "v2_pipeline"

train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VALIDATION_FILE)

X_train = train.drop(columns=["Index", "demand"])
y_train = train["demand"]

X_val = val.drop(columns=["Index", "demand"])
y_val = val["demand"]

params = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "random_state": 42,
}

model = LGBMRegressor(**params)

start_time = time.perf_counter()
model.fit(X_train, y_train)
train_time_seconds = time.perf_counter() - start_time

preds = model.predict(X_val)

mae = mean_absolute_error(y_val, preds)
rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))

print("MAE:", mae)
print("RMSE:", rmse)

dataset_info = {
    "train_file": TRAIN_FILE,
    "validation_file": VALIDATION_FILE,
    "train_rows": len(train),
    "validation_rows": len(val),
    "num_features": X_train.shape[1],
    "feature_columns": list(X_train.columns),
}

log_experiment(
    model_name="LightGBM",
    metrics={"mae": mae, "rmse": rmse},
    hyperparameters=params,
    dataset_info=dataset_info,
    feature_set_version=FEATURE_SET_VERSION,
    feature_pipeline_name="feature_engineering.py",
    feature_pipeline_notes="20-feature target encoding, lag, rolling, temporal, spatial, weather, and interaction pipeline",
    train_time_seconds=train_time_seconds,
)
