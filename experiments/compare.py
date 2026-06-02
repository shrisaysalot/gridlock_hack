from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"


def main() -> None:
    results = _load_results()
    if results.empty:
        print("No scored experiments logged yet.")
        return

    print(f"Best MAE: {results['mae'].min()}")
    print(f"Worst MAE: {results['mae'].max()}")
    print(f"Average MAE: {results['mae'].mean()}")

    print("\nParameter correlations with MAE")
    correlations = _parameter_correlations(results)
    if correlations.empty:
        print("Not enough numeric hyperparameter variation to compute correlations.")
    else:
        print(correlations.to_string(index=False))

    print("\nPerformance grouped by feature set")
    grouped = (
        results.groupby("feature_set_version")["mae"]
        .agg(["count", "min", "mean", "max"])
        .sort_values("min", ascending=True)
        .reset_index()
    )
    print(grouped.to_string(index=False))


def _load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        return pd.DataFrame()
    results = pd.read_csv(RESULTS_CSV)
    if "mae" not in results.columns:
        return pd.DataFrame()
    results["mae"] = pd.to_numeric(results["mae"], errors="coerce")
    return results.dropna(subset=["mae"])


def _parameter_correlations(results: pd.DataFrame) -> pd.DataFrame:
    params = []
    for _, row in results.iterrows():
        try:
            params.append(json.loads(row.get("hyperparameters_json", "{}")))
        except json.JSONDecodeError:
            params.append({})

    param_frame = pd.DataFrame(params)
    if param_frame.empty:
        return pd.DataFrame()

    numeric_params = param_frame.apply(pd.to_numeric, errors="coerce")
    numeric_params["mae"] = results["mae"].to_numpy()
    correlations = []
    for column in numeric_params.columns:
        if column == "mae":
            continue
        series = numeric_params[column]
        valid = pd.DataFrame({"param": series, "mae": numeric_params["mae"]}).dropna()
        if len(valid) < 3 or valid["param"].nunique() < 2:
            continue
        corr = valid["param"].corr(valid["mae"])
        if np.isfinite(corr):
            correlations.append({"parameter": column, "correlation_with_mae": corr})
    correlation_frame = pd.DataFrame(correlations)
    if correlation_frame.empty:
        return correlation_frame
    return correlation_frame.sort_values("correlation_with_mae", key=lambda s: s.abs(), ascending=False)


if __name__ == "__main__":
    main()
