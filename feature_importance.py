import pandas as pd
from xgboost import XGBRegressor

train = pd.read_csv("train_features_v2.csv")

X = train.drop(columns=["Index", "demand"])
y = train["demand"]

model = XGBRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

imp = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

print(
    imp.sort_values("importance", ascending=False)
       .head(20)
)