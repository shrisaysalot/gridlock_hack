# Gridlock Experiment Tracking

This repository uses a lightweight, file-based experiment tracker in `experiments/`.
There are no external tracking services and no MLflow dependency.

## Automatic Training Logs

Training scripts call:

```python
from experiments.logger import log_experiment

log_experiment(
    model_name="XGBoost",
    metrics={"mae": mae},
    hyperparameters=params,
    dataset_info=dataset_info,
    feature_set_version="v2_pipeline",
)
```

Each completed run appends one row to `experiments/results.csv`. The logger records:

- run metadata: experiment ID, timestamp, git commit, branch, username, hostname, model, script
- dataset metadata: file names, row counts, feature count, plus the full dataset metadata JSON
- feature metadata: feature set version, pipeline name, and notes
- metrics: MAE, RMSE when supplied, and train time
- hyperparameters: common columns plus every additional model-specific parameter
- JSON snapshots: `hyperparameters_json` and `dataset_metadata_json`

CSV writes are protected with a lock file and written atomically, which makes local concurrent runs reasonably safe.

## BEST_MODEL.md

`BEST_MODEL.md` is updated automatically whenever a new run beats the current best MAE in `experiments/results.csv`.

It includes the best MAE, timestamp, model name, feature set version, full hyperparameter configuration, training script, git commit hash, and experiment ID.

## Viewing Results

Run the leaderboard:

```bash
python experiments/leaderboard.py
```

This prints the top 20 experiments by MAE, the best run per model family, the best run per feature set, and the total number of logged experiments.

Compare experiments:

```bash
python experiments/compare.py
```

This prints the best, worst, and average MAE, parameter correlations with MAE when enough numeric data exists, and grouped feature-set performance.

Print best model configs:

```bash
python experiments/best_configs.py
```

This prints the best XGBoost, LightGBM, CatBoost, and overall runs with MAE, feature set, experiment ID, and key hyperparameters.

## Contributing New Runs

Before pushing experiment results:

1. Pull the latest repository changes.
2. Run your training script.
3. Confirm `experiments/results.csv` has one new row.
4. Check `BEST_MODEL.md`; it changes only if your MAE is the new best.
5. Commit your training changes, `experiments/results.csv`, and `BEST_MODEL.md` when relevant.

Do not commit raw datasets, generated submissions, model checkpoints, cache directories, or notebook checkpoints.

## Creating a New Feature Set Version

When changing feature engineering:

1. Choose a clear feature set version, for example `old_pipeline`, `v2_pipeline`, or `geohash_target_encoding_v3`.
2. Save train and validation feature files with names that match the version.
3. Update the training script constants:

```python
TRAIN_FILE = "train_features_v3.csv"
VALIDATION_FILE = "val_features_v3.csv"
FEATURE_SET_VERSION = "geohash_target_encoding_v3"
```

4. Set `feature_pipeline_name` and `feature_pipeline_notes` in the `log_experiment(...)` call.

This makes feature-set comparisons available in `experiments/leaderboard.py` and `experiments/compare.py`.

## Prediction Artifact Tracking

When a submission or prediction file is generated, log it with:

```python
from experiments.logger import log_prediction_artifact

log_prediction_artifact(
    prediction_file_name="submissions/xgb_v2.csv",
    model_used="XGBoost",
    feature_set_version="v2_pipeline",
    local_validation_mae=mae,
    experiment_id=experiment_id,
)
```

If the prediction file is produced as part of training, you can also pass `prediction_info` to `log_experiment(...)` so the prediction metadata is stored on the experiment row.

## Future Sweeps

The logging interface is stable for manual runs, grid search, Bayesian optimization, hyperparameter sweeps, and Optuna. Sweep code should call `log_experiment(...)` once per trained configuration with the same arguments used by single-run scripts.
