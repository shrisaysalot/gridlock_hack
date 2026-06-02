from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"


def main() -> None:
    results = _load_results()
    if results.empty:
        print("No experiments logged yet.")
        return

    print(f"Total experiments logged: {len(results)}")
    print("\nTop 20 experiments by MAE")
    print(_format_table(results.sort_values("mae", ascending=True).head(20)))

    print("\nBest experiment per model family")
    by_model = results.loc[results.groupby("model_name")["mae"].idxmin()].sort_values("mae")
    print(_format_table(by_model))

    print("\nBest experiment per feature set")
    by_feature = results.loc[results.groupby("feature_set_version")["mae"].idxmin()].sort_values("mae")
    print(_format_table(by_feature))


def _load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        return pd.DataFrame()
    results = pd.read_csv(RESULTS_CSV)
    if "mae" not in results.columns:
        return pd.DataFrame()
    results["mae"] = pd.to_numeric(results["mae"], errors="coerce")
    return results.dropna(subset=["mae"])


def _format_table(frame: pd.DataFrame) -> str:
    columns = [
        "mae",
        "model_name",
        "feature_set_version",
        "script_name",
        "timestamp",
        "experiment_id",
    ]
    visible = [column for column in columns if column in frame.columns]
    return frame[visible].to_string(index=False)


if __name__ == "__main__":
    main()
