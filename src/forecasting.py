import math
import time
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ACTIVE_THRESHOLD

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

TEST_DAYS = 30
FEATURE_COLS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "roll_mean_7",
    "roll_std_7",
    "trend_7",
    "vaccination_indicator",
    "day_of_week",
    "month",
    "day_of_year",
]


def add_features(cdf: pd.DataFrame) -> pd.DataFrame:
    cdf = cdf.sort_values("date").copy()

    for lag in [1, 2, 3, 7, 14]:
        cdf[f"lag_{lag}"] = cdf["cases_per_100k_7d"].shift(lag)

    cdf["roll_mean_7"] = cdf["cases_per_100k_7d"].shift(1).rolling(7, min_periods=3).mean()
    cdf["roll_std_7"] = cdf["cases_per_100k_7d"].shift(1).rolling(7, min_periods=3).std()
    cdf["trend_7"] = cdf["lag_1"] - cdf["lag_7"]

    cdf["day_of_week"] = cdf["date"].dt.dayofweek
    cdf["month"] = cdf["date"].dt.month
    cdf["day_of_year"] = cdf["date"].dt.dayofyear

    cdf["vaccination_indicator"] = pd.to_numeric(
        cdf["vaccination_indicator"], errors="coerce"
    ).interpolate(limit_direction="both")

    cdf["roll_std_7"] = cdf["roll_std_7"].fillna(0)
    return cdf



def safe_mape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100



def smape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, eps)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100



def evaluate_forecast(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": safe_mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }



def forecast_naive(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    last_value = float(train_df["cases_per_100k_7d"].iloc[-1])
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = np.full(len(test_df), last_value)
    pred_time = time.perf_counter() - t1
    return preds, train_time, pred_time



def forecast_moving_average(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window: int = 7,
) -> Tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    history = list(train_df["cases_per_100k_7d"].astype(float).values)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = []
    for _ in range(len(test_df)):
        pred = float(np.mean(history[-window:]))
        preds.append(pred)
        history.append(pred)
    pred_time = time.perf_counter() - t1
    return np.array(preds), train_time, pred_time



def forecast_random_forest_recursive(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Iterable[str] = FEATURE_COLS,
) -> Tuple[np.ndarray, float, float]:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train = train_df[list(feature_cols)]
    y_train = train_df["cases_per_100k_7d"]

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    history = train_df.copy().reset_index(drop=True)
    preds = []

    t1 = time.perf_counter()
    for i in range(len(test_df)):
        row = test_df.iloc[[i]].copy()

        row["lag_1"] = history["cases_per_100k_7d"].iloc[-1]
        row["lag_2"] = history["cases_per_100k_7d"].iloc[-2]
        row["lag_3"] = history["cases_per_100k_7d"].iloc[-3]
        row["lag_7"] = history["cases_per_100k_7d"].iloc[-7]
        row["lag_14"] = history["cases_per_100k_7d"].iloc[-14]
        row["roll_mean_7"] = history["cases_per_100k_7d"].iloc[-7:].mean()
        row["roll_std_7"] = history["cases_per_100k_7d"].iloc[-7:].std()
        row["trend_7"] = history["cases_per_100k_7d"].iloc[-1] - history["cases_per_100k_7d"].iloc[-7]

        pred = float(model.predict(row[list(feature_cols)])[0])
        pred = max(pred, 0.0)
        preds.append(pred)

        new_row = row.copy()
        new_row["cases_per_100k_7d"] = pred
        history = pd.concat([history, new_row], ignore_index=True)

    pred_time = time.perf_counter() - t1
    return np.array(preds), train_time, pred_time



def forecast_arima(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[np.ndarray, float, float]:
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is not available")

    y_train = train_df["cases_per_100k_7d"].astype(float).to_numpy()
    if len(y_train) < 20:
        raise ValueError("ARIMA requires at least 20 training observations")

    t0 = time.perf_counter()
    model = ARIMA(
        y_train,
        order=order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    fitted = model.fit()
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = np.asarray(fitted.forecast(steps=len(test_df)), dtype=float)
    preds = np.nan_to_num(preds, nan=0.0, posinf=np.nanmax(y_train), neginf=0.0)
    preds = np.clip(preds, 0.0, None)
    pred_time = time.perf_counter() - t1
    return preds, train_time, pred_time



def run_forecasting(df: pd.DataFrame, test_days: int = TEST_DAYS):
    results = []
    forecast_plots = {}

    for country in sorted(df["country"].unique()):
        cdf = df[df["country"] == country].copy()
        cdf = add_features(cdf)
        cdf = cdf.dropna(subset=["cases_per_100k_7d"]).reset_index(drop=True)

        if len(cdf) <= test_days + 20:
            print(f"{country} skipped: not enough rows for a {test_days}-day forecast.")
            continue

        active_df = cdf[cdf["cases_per_100k_7d"] > ACTIVE_THRESHOLD].copy()

        if len(active_df) <= test_days + 20:
            print(f"{country} skipped: not enough active data.")
            continue

        train_df = active_df.iloc[:-test_days].copy()
        test_df = active_df.iloc[-test_days:].copy()

        train_df = train_df.dropna(subset=FEATURE_COLS + ["cases_per_100k_7d"]).reset_index(drop=True)
        test_df = test_df.dropna(subset=["cases_per_100k_7d"]).reset_index(drop=True)

        if len(train_df) < 20 or len(test_df) == 0:
            print(f"{country} skipped: not enough usable rows after feature construction.")
            continue

        y_test = test_df["cases_per_100k_7d"].to_numpy()

        model_functions = {
            "Naive": lambda: forecast_naive(train_df, test_df),
            "Moving Average": lambda: forecast_moving_average(train_df, test_df, window=7),
            "Random Forest": lambda: forecast_random_forest_recursive(train_df, test_df, FEATURE_COLS),
        }

        if HAS_STATSMODELS:
            model_functions["ARIMA"] = lambda: forecast_arima(train_df, test_df, order=(1, 1, 1))

        for model_name, model_fn in model_functions.items():
            try:
                preds, train_time, pred_time = model_fn()
                metrics = evaluate_forecast(y_test, preds)
                results.append(
                    {
                        "country": country,
                        "model": model_name,
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "MAPE": metrics["MAPE"],
                        "sMAPE": metrics["sMAPE"],
                        "train_seconds": train_time,
                        "predict_seconds": pred_time,
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                    }
                )
                forecast_plots[(country, model_name)] = pd.DataFrame(
                    {
                        "date": test_df["date"].values,
                        "actual": y_test,
                        "predicted": preds,
                    }
                )
            except Exception as exc:
                print(f"{country} - {model_name} failed: {exc}")

    forecast_results = pd.DataFrame(results)
    if not forecast_results.empty:
        forecast_results = forecast_results.sort_values(["country", "RMSE"]).reset_index(drop=True)
        best_models = (
            forecast_results.sort_values(["country", "RMSE"])
            .groupby("country", as_index=False)
            .first()[["country", "model", "RMSE"]]
        )
    else:
        best_models = pd.DataFrame(columns=["country", "model", "RMSE"])

    return forecast_results, best_models, forecast_plots