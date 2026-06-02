"""
Feature Engineering Pipeline — All 20 Features
================================================
Tier 1: Target encoding + lag/rolling features
Tier 2: Temporal cyclical + flags
Tier 3: Spatial & weather
Tier 4: Interaction features

Fixes applied:
  1. Target encodings fit on train only, mapped onto val — no leakage.
  2. Lag/rolling NaN fill uses per-geohash train mean, not current demand.
  3. zone_mean_demand computed from train only.
  4. day_of_week anchored via DAY_ONE_WEEKDAY config constant.
  5. time_of_day vectorised with pd.cut (no row-wise .apply).
  6. road_capacity vectorised with np.where (no row-wise .apply).
  BUGFIX: Replaced pivot+stack lag approach with groupby().shift() to
          prevent row explosion on sparse (geohash, slot) data.

Usage:
    python feature_engineering.py

Outputs:
    train_features.csv
    val_features.csv
"""

import numpy as np
import pandas as pd
import geohash2
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# FIX 4: weekday of day=1 in your dataset. 0=Monday … 6=Sunday.
DAY_ONE_WEEKDAY = 0

# ─────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("train.csv")
# ─────────────────────────────────────────────
# 1. PARSE TIMESTAMP → hour, minute, slot
# ─────────────────────────────────────────────
print("Parsing timestamps...")

df[["hour", "minute"]] = df["timestamp"].str.split(":", expand=True).astype(int)
df["slot"]        = df["hour"] * 4 + df["minute"] // 15
df["day_of_week"] = (df["day"] - 1 + DAY_ONE_WEEKDAY) % 7   # FIX 4

# Absolute slot for chronological ordering across days
df["abs_slot"] = (df["day"] - df["day"].min()) * 96 + df["slot"]

# Sort chronologically within each geohash
df = df.sort_values(["geohash", "abs_slot"]).reset_index(drop=True)

# ─────────────────────────────────────────────
# 2. TRAIN / VAL SPLIT — must happen BEFORE any target aggregation
# ─────────────────────────────────────────────
last_day = df["day"].max()
train_df = df[df["day"] < last_day].copy()
val_df   = df[df["day"] == last_day].copy()

global_mean = train_df["demand"].mean()

# ─────────────────────────────────────────────
# 3. TIER 1 — LAG + ROLLING (groupby shift — no pivot, no row explosion)
# ─────────────────────────────────────────────
print("Computing lag and rolling features...")

# Work on the full df sorted correctly so val rows see train lags
df_sorted = df.sort_values(["geohash", "abs_slot"]).copy()

df_sorted["lag1_demand"]    = df_sorted.groupby("geohash")["demand"].shift(1)
df_sorted["lag4_demand"]    = df_sorted.groupby("geohash")["demand"].shift(4)
df_sorted["rolling_mean_4"] = (
    df_sorted.groupby("geohash")["demand"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
)
df_sorted["rolling_std_4"]  = (
    df_sorted.groupby("geohash")["demand"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).std())
).fillna(0)

# Merge lag cols back onto train/val by Index
lag_cols = ["Index", "lag1_demand", "lag4_demand", "rolling_mean_4", "rolling_std_4"]
lag_df   = df_sorted[lag_cols]

train_df = train_df.merge(lag_df, on="Index", how="left")
val_df   = val_df.merge(lag_df,   on="Index", how="left")

# FIX 2: NaN fallback = per-geohash train mean (not current demand)
geo_mean_demand = train_df.groupby("geohash")["demand"].mean()
for col in ["lag1_demand", "lag4_demand", "rolling_mean_4"]:
    for split in [train_df, val_df]:
        split[col] = split[col].fillna(split["geohash"].map(geo_mean_demand)).fillna(global_mean)

# ─────────────────────────────────────────────
# 4. TIER 1 — TARGET ENCODING (train only → map to val)
# ─────────────────────────────────────────────
print("Building target encodings...")

def apply_target_enc(train, val, group_cols, target_col, enc_name, fallback):
    enc = train.groupby(group_cols)[target_col].mean().rename(enc_name)
    for split in [train, val]:
        split[enc_name] = split.set_index(group_cols).index.map(enc).values
        split[enc_name] = split[enc_name].fillna(fallback)

apply_target_enc(train_df, val_df, ["geohash", "slot"],        "demand", "geo_slot_target_enc",    global_mean)
apply_target_enc(train_df, val_df, ["geohash", "day_of_week"], "demand", "geo_dow_target_enc",     global_mean)

# ─────────────────────────────────────────────
# 5. TIER 2 — TEMPORAL CYCLICAL FEATURES
# ─────────────────────────────────────────────
print("Building temporal features...")

for split in [train_df, val_df]:
    split["hour_sin"]    = np.sin(2 * np.pi * split["hour"] / 24)
    split["hour_cos"]    = np.cos(2 * np.pi * split["hour"] / 24)
    split["slot_sin"]    = np.sin(2 * np.pi * split["slot"] / 96)
    split["slot_cos"]    = np.cos(2 * np.pi * split["slot"] / 96)
    split["is_rush_hour"] = (
        ((split["slot"] >= 28) & (split["slot"] <= 35)) |
        ((split["slot"] >= 68) & (split["slot"] <= 75))
    ).astype(int)
    split["is_weekend"] = (split["day_of_week"] >= 5).astype(int)
    # FIX 5: vectorised pd.cut instead of row-wise apply
    split["time_of_day"] = pd.cut(
        split["hour"],
        bins=[-1, 5, 11, 16, 23],
        labels=[0, 1, 2, 3]        # Night, Morning, Afternoon, Evening
    ).astype(float).astype(int)

# ─────────────────────────────────────────────
# 6. TIER 3 — SPATIAL & WEATHER FEATURES
# ─────────────────────────────────────────────
print("Building spatial and weather features...")

# FIX 3: zone_mean from train only
for split in [train_df, val_df]:
    split["geo_prefix"] = split["geohash"].str[:4]

zone_mean = train_df.groupby("geo_prefix")["demand"].mean().rename("zone_mean_demand")
for split in [train_df, val_df]:
    split["zone_mean_demand"] = split["geo_prefix"].map(zone_mean).fillna(global_mean)

# Lat/lon from geohash
print("  Decoding geohashes...")
def decode_geohash(gh):
    try:
        lat, lon, _, _ = geohash2.decode_exactly(gh)
        return float(lat), float(lon)
    except Exception:
        return np.nan, np.nan

all_geohashes = pd.concat([train_df["geohash"], val_df["geohash"]]).unique()
geo_latlon    = {gh: decode_geohash(gh) for gh in all_geohashes}

for split in [train_df, val_df]:
    split["lat"] = split["geohash"].map(lambda g: geo_latlon[g][0])
    split["lon"] = split["geohash"].map(lambda g: geo_latlon[g][1])

# Weather
weather_map    = {"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3}
weather_median = float(pd.Series(weather_map).median())

for split in [train_df, val_df]:
    split["weather_severity"] = split["Weather"].map(weather_map).fillna(weather_median)
    split["bad_weather"]      = split["Weather"].isin(["Rainy", "Snowy"]).astype(int)

# Temperature bin — mode computed from train only
temp_bins       = [-np.inf, 0, 10, 20, 30, np.inf]
temp_labels     = [0, 1, 2, 3, 4]
train_temp_mode = float(
    pd.cut(train_df["Temperature"], bins=temp_bins, labels=temp_labels).mode()[0]
)
for split in [train_df, val_df]:
    split["temp_bin"] = pd.cut(
        split["Temperature"], bins=temp_bins, labels=temp_labels
    ).astype(float).fillna(train_temp_mode)

# ─────────────────────────────────────────────
# 7. TIER 4 — INTERACTION FEATURES
# ─────────────────────────────────────────────
print("Building interaction features...")

for split in [train_df, val_df]:
    split["rush_x_bad_weather"] = split["is_rush_hour"] * split["bad_weather"]
    split["cold_snow"] = (
        (split["Temperature"] < 5) & (split["Weather"] == "Snowy")
    ).fillna(False).astype(int)

# FIX 1d: geo × weather severity — train only
apply_target_enc(train_df, val_df, ["geohash", "weather_severity"], "demand", "geo_weather_target_enc", global_mean)

# FIX 6: road_capacity vectorised with np.where
for split in [train_df, val_df]:
    split["road_capacity"] = np.where(
        split["LargeVehicles"] == "Allowed",
        split["NumberofLanes"] * 1.5,
        split["NumberofLanes"]
    )

# ─────────────────────────────────────────────
# 8. FEATURE COLUMN LIST
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "geo_slot_target_enc",   # 1
    "geo_dow_target_enc",    # 2
    "lag1_demand",           # 3
    "lag4_demand",           # 4
    "rolling_mean_4",        # 5
    "hour_sin",              # 6
    "hour_cos",
    "slot_sin",              # 7
    "slot_cos",
    "is_rush_hour",          # 8
    "is_weekend",            # 9
    "time_of_day",           # 10
    "zone_mean_demand",      # 11
    "lat",                   # 12
    "lon",
    "weather_severity",      # 13
    "bad_weather",           # 14
    "temp_bin",              # 15
    "rush_x_bad_weather",    # 16
    "cold_snow",             # 17
    "geo_weather_target_enc",# 18
    "road_capacity",         # 19
    "rolling_std_4",         # 20
]

# ─────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────
OUTPUT_COLS = ["Index"] + FEATURE_COLS + ["demand"]
train_df[OUTPUT_COLS].to_csv("train_features_v2.csv", index=False)
val_df[OUTPUT_COLS].to_csv("val_features_v2.csv", index=False)

print(f"\nDataset summary")
print(f"  Train rows : {len(train_df):,}  (days < {last_day})")
print(f"  Val rows   : {len(val_df):,}  (day == {last_day})")
print(f"  Features   : {len(FEATURE_COLS)}")

# ─────────────────────────────────────────────
# 10. SANITY CHECK
# ─────────────────────────────────────────────
print("\nNull counts in train features:")
null_summary = train_df[FEATURE_COLS].isnull().sum()
print(null_summary[null_summary > 0].to_string() if null_summary.sum() > 0 else "  ✓ No nulls")

print("\nLeakage check — lag1 == demand (should be ~0%):")
leaky = (train_df["lag1_demand"] == train_df["demand"]).sum()
print(f"  {leaky} / {len(train_df)} ({100*leaky/len(train_df):.2f}%)")