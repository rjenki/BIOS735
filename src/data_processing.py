import numpy as np
import pandas as pd

from .config import (
    ACTIVE_THRESHOLD,
    COUNTRIES,
    DATA_PATH,
    MAX_REPORTING_GAP_DAYS,
)

def load_raw_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["country", "date", "new_cases", "population"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df = df[df["country"].isin(COUNTRIES)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["country", "date"]).reset_index(drop=True)

    df["new_cases"] = pd.to_numeric(df["new_cases"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["population"])
    df = df[df["population"] > 0].copy()

    # Track whether the original row had a reported case count before any interpolation.
    df["new_cases_reported"] = df["new_cases"].notna()
    df.loc[df["new_cases"] < 0, "new_cases"] = np.nan

    vaccination_candidates = [
        "people_fully_vaccinated_per_hundred",
        "people_vaccinated_per_hundred",
        "total_vaccinations_per_hundred",
    ]
    available_vax = [c for c in vaccination_candidates if c in df.columns]
    if available_vax:
        df["vaccination_indicator"] = df[available_vax].bfill(axis=1).iloc[:, 0]
    else:
        df["vaccination_indicator"] = np.nan

    df["vaccination_indicator"] = (
        pd.to_numeric(df["vaccination_indicator"], errors="coerce")
        .groupby(df["country"])
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )

    trimmed = []
    for country in df["country"].unique():
        cdf = df[df["country"] == country].copy()
        reported_mask = cdf["new_cases_reported"]
        if reported_mask.any():
            first_idx = reported_mask.idxmax()
            last_idx = reported_mask[::-1].idxmax()
            cdf = cdf.loc[first_idx:last_idx].copy()
            trimmed.append(cdf)

    if not trimmed:
        return pd.DataFrame()

    df = pd.concat(trimmed, ignore_index=True)

    # Interpolate within each country. Do not convert remaining gaps to zeros,
    # because that creates artificial flatlines and can produce fake perfect forecasts.
    df["new_cases"] = (
        df.groupby("country")["new_cases"]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )
    df = _compute_case_rates(df)

    # Keep the full observed span for each country once trimming has bounded the series.
    # The older "longest contiguous reporting block" logic could discard valid late-period
    # data for countries with irregular reporting and was the main cause of the Japan issue.
    span_trimmed = []
    for country in df["country"].unique():
        cdf = df[df["country"] == country].copy().sort_values("date").reset_index(drop=True)
        if cdf["new_cases_reported"].any():
            start_date = cdf.loc[cdf["new_cases_reported"], "date"].min()
            end_date = cdf.loc[cdf["new_cases_reported"], "date"].max()
            cdf = cdf[(cdf["date"] >= start_date) & (cdf["date"] <= end_date)].copy()
            span_trimmed.append(cdf)

    if not span_trimmed:
        return pd.DataFrame()

    df = pd.concat(span_trimmed, ignore_index=True)

    # Recompute after final trimming.
    df["new_cases"] = (
        df.groupby("country")["new_cases"]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )
    df = _compute_case_rates(df)

    trimmed_active = []
    for country in df['country'].unique():
        cdf = df[df['country'] == country].copy().sort_values('date').reset_index(drop=True)

        active_mask = cdf['cases_per_100k_7d'] > ACTIVE_THRESHOLD
        if active_mask.any():
            first_pos = np.where(active_mask)[0][0]
            last_pos = np.where(active_mask)[0][-1]
            cdf = cdf.iloc[first_pos:last_pos + 1].copy()

        trimmed_active.append(cdf)

    df = pd.concat(trimmed_active, ignore_index=True)

    df = df[df['date'] <= pd.Timestamp("2023-06-30")].copy()

    return df.sort_values(['country', 'date']).reset_index(drop=True)

def _compute_case_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cases_per_100k"] = (df["new_cases"] / df["population"]) * 100000
    df["cases_per_100k"] = (
        df.groupby("country")["cases_per_100k"]
        .transform(lambda s: s.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both"))
    )
    df["cases_per_100k_7d"] = (
        df.groupby("country")["cases_per_100k"]
        .transform(lambda s: s.rolling(window=7, min_periods=7).mean())
    )
    return df

def summarize_coverage(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("country")
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            rows=("date", "count"),
        )
        .reset_index()
    )