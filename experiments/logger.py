from __future__ import annotations

import csv
import getpass
import inspect
import json
import math
import os
import platform
import socket
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
PREDICTIONS_CSV = EXPERIMENTS_DIR / "predictions.csv"
BEST_MODEL_MD = ROOT_DIR / "BEST_MODEL.md"
LOCK_FILE = EXPERIMENTS_DIR / ".results.lock"

BASE_COLUMNS = [
    "experiment_id",
    "timestamp",
    "git_commit_hash",
    "git_branch",
    "username",
    "hostname",
    "model_name",
    "script_name",
    "train_file",
    "validation_file",
    "train_rows",
    "validation_rows",
    "num_features",
    "feature_set_version",
    "feature_pipeline_name",
    "feature_pipeline_notes",
    "mae",
    "rmse",
    "train_time_seconds",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "hyperparameters_json",
    "dataset_metadata_json",
    "prediction_file_name",
    "prediction_timestamp",
    "model_used",
    "local_validation_mae",
]

PREDICTION_COLUMNS = [
    "prediction_file_name",
    "prediction_timestamp",
    "model_used",
    "feature_set_version",
    "local_validation_mae",
    "experiment_id",
]


def log_experiment(
    model_name: str,
    metrics: dict[str, Any],
    hyperparameters: dict[str, Any] | None = None,
    dataset_info: dict[str, Any] | None = None,
    feature_set_version: str = "",
    feature_pipeline_name: str = "",
    feature_pipeline_notes: str = "",
    train_time_seconds: float | None = None,
    script_name: str | None = None,
    prediction_info: dict[str, Any] | None = None,
) -> str:
    """Append one experiment row and update BEST_MODEL.md when MAE improves.

    The public interface is intentionally small so sweeps, Optuna, and future
    model families can keep calling this function with the same arguments.
    """

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    experiment_id = str(uuid.uuid4())
    timestamp = _utc_now()
    hyperparameters = hyperparameters or {}
    dataset_info = dataset_info or {}
    metrics = metrics or {}
    prediction_info = prediction_info or {}

    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "git_commit_hash": _git_value(["git", "rev-parse", "--short", "HEAD"]),
        "git_branch": _git_value(["git", "branch", "--show-current"]),
        "username": _username(),
        "hostname": socket.gethostname(),
        "model_name": model_name,
        "script_name": script_name or _calling_script_name(),
        "train_file": dataset_info.get("train_file", ""),
        "validation_file": dataset_info.get("validation_file", ""),
        "train_rows": dataset_info.get("train_rows", ""),
        "validation_rows": dataset_info.get("validation_rows", ""),
        "num_features": dataset_info.get("num_features", ""),
        "feature_set_version": feature_set_version,
        "feature_pipeline_name": feature_pipeline_name,
        "feature_pipeline_notes": feature_pipeline_notes,
        "mae": metrics.get("mae", ""),
        "rmse": metrics.get("rmse", ""),
        "train_time_seconds": train_time_seconds if train_time_seconds is not None else metrics.get("train_time_seconds", ""),
        "hyperparameters_json": _json_dumps(hyperparameters),
        "dataset_metadata_json": _json_dumps(dataset_info),
        "prediction_file_name": prediction_info.get("prediction_file_name", ""),
        "prediction_timestamp": prediction_info.get("prediction_timestamp", ""),
        "model_used": prediction_info.get("model_used", model_name if prediction_info else ""),
        "local_validation_mae": prediction_info.get("local_validation_mae", metrics.get("mae", "") if prediction_info else ""),
    }

    for key in ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree"]:
        row[key] = hyperparameters.get(key, "")

    for key, value in hyperparameters.items():
        if key not in row:
            row[key] = _stringify(value)

    with _file_lock(LOCK_FILE):
        _append_row_with_dynamic_columns(RESULTS_CSV, row, BASE_COLUMNS)
        _update_best_model()

    return experiment_id


def log_prediction_artifact(
    prediction_file_name: str,
    model_used: str,
    feature_set_version: str,
    local_validation_mae: float | None = None,
    experiment_id: str | None = None,
) -> None:
    """Track a generated submission or prediction artifact."""

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "prediction_file_name": prediction_file_name,
        "prediction_timestamp": _utc_now(),
        "model_used": model_used,
        "feature_set_version": feature_set_version,
        "local_validation_mae": local_validation_mae if local_validation_mae is not None else "",
        "experiment_id": experiment_id or "",
    }
    with _file_lock(LOCK_FILE):
        _append_row_with_dynamic_columns(PREDICTIONS_CSV, row, PREDICTION_COLUMNS)


def _append_row_with_dynamic_columns(path: Path, row: dict[str, Any], base_columns: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        columns = list(existing.columns)
    else:
        existing = pd.DataFrame()
        columns = []

    for column in base_columns:
        if column not in columns:
            columns.append(column)
    for column in row:
        if column not in columns:
            columns.append(column)

    new_row = {column: _stringify(row.get(column, "")) for column in columns}
    updated = pd.concat([existing.reindex(columns=columns), pd.DataFrame([new_row], columns=columns)], ignore_index=True)
    _atomic_write_csv(path, updated, columns)


def _update_best_model() -> None:
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        _write_no_best_model()
        return

    results = pd.read_csv(RESULTS_CSV)
    if "mae" not in results.columns:
        _write_no_best_model()
        return

    results["mae_numeric"] = pd.to_numeric(results["mae"], errors="coerce")
    scored = results.dropna(subset=["mae_numeric"])
    if scored.empty:
        _write_no_best_model()
        return

    best = scored.sort_values("mae_numeric", ascending=True).iloc[0]
    hyperparameters = _load_json(best.get("hyperparameters_json", "{}"))
    lines = [
        "# Best Model",
        "",
        f"- current best MAE: {best['mae_numeric']}",
        f"- timestamp achieved: {best.get('timestamp', '')}",
        f"- model name: {best.get('model_name', '')}",
        f"- feature set version: {best.get('feature_set_version', '')}",
        f"- training script used: {best.get('script_name', '')}",
        f"- git commit hash: {best.get('git_commit_hash', '')}",
        f"- experiment ID: {best.get('experiment_id', '')}",
        "",
        "## Full Hyperparameter Configuration",
        "",
        "```json",
        json.dumps(hyperparameters, indent=2, sort_keys=True),
        "```",
        "",
    ]
    _atomic_write_text(BEST_MODEL_MD, "\n".join(lines))


def _write_no_best_model() -> None:
    _atomic_write_text(BEST_MODEL_MD, "# Best Model\n\nNo scored experiments have been logged yet.\n")


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock:
        if platform.system() != "Windows":
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        else:
            yield


def _atomic_write_csv(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8", newline="") as handle:
        temp_path = Path(handle.name)
        frame.to_csv(handle, index=False, columns=columns, quoting=csv.QUOTE_MINIMAL)
    os.replace(temp_path, path)


def _atomic_write_text(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
        temp_path = Path(handle.name)
        handle.write(text)
    os.replace(temp_path, path)


def _git_value(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT_DIR, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _username() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "")


def _calling_script_name() -> str:
    for frame in inspect.stack():
        filename = Path(frame.filename)
        if filename.name != Path(__file__).name:
            return filename.name
    return ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _load_json(value: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) and value else {}
    except json.JSONDecodeError:
        return {}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value)
    return str(value)
