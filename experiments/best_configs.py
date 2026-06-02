from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"
MODEL_FAMILIES = ["XGBoost", "LightGBM", "CatBoost"]
KEY_PARAMS = [
    "n_estimators",
    "iterations",
    "learning_rate",
    "max_depth",
    "depth",
    "num_leaves",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
]


def main() -> None:
    results = _load_results()
    if results.empty:
        print("No scored experiments logged yet.")
        return

    for family in MODEL_FAMILIES:
        print(f"\nBest {family} run")
        family_rows = results[results["model_name"].str.lower().str.contains(family.lower(), na=False)]
        _print_best(family_rows)

    print("\nBest overall run")
    _print_best(results)


def _load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        return pd.DataFrame()
    results = pd.read_csv(RESULTS_CSV)
    if "mae" not in results.columns:
        return pd.DataFrame()
    results["mae"] = pd.to_numeric(results["mae"], errors="coerce")
    return results.dropna(subset=["mae"])


def _print_best(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("No runs logged.")
        return
    best = frame.sort_values("mae", ascending=True).iloc[0]
    params = _load_params(best.get("hyperparameters_json", "{}"))
    key_params = {key: params[key] for key in KEY_PARAMS if key in params}
    print(f"MAE: {best.get('mae', '')}")
    print(f"Feature set: {best.get('feature_set_version', '')}")
    print(f"Experiment ID: {best.get('experiment_id', '')}")
    print(f"Key hyperparameters: {json.dumps(key_params, sort_keys=True)}")


def _load_params(value: str) -> dict:
    try:
        return json.loads(value) if value else {}
    except json.JSONDecodeError:
        return {}


if __name__ == "__main__":
    main()
