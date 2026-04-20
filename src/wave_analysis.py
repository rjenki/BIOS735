import numpy as np
import pandas as pd

from .config import COUNTRIES, MIN_WAVE_DAYS, SENSITIVITY_THRESHOLDS, WAVE_THRESHOLD_PER_100K


def summarize_wave(country_df: pd.DataFrame, start_idx: int, end_idx: int, wave_id: int, previous_end=None) -> dict:
    segment = country_df.iloc[start_idx:end_idx + 1].copy()
    peak_local = int(segment['cases_per_100k_7d'].to_numpy().argmax())
    peak_idx = start_idx + peak_local

    start_date = country_df.iloc[start_idx]['date']
    peak_date = country_df.iloc[peak_idx]['date']
    end_date = country_df.iloc[end_idx]['date']

    peak_height = float(country_df.iloc[peak_idx]['cases_per_100k_7d'])
    duration_days = int((end_date - start_date).days + 1)

    start_value = float(country_df.iloc[start_idx]['cases_per_100k_7d'])
    end_value = float(country_df.iloc[end_idx]['cases_per_100k_7d'])

    rise_days = max((peak_date - start_date).days, 1)
    fall_days = max((end_date - peak_date).days, 1)

    growth_rate = (peak_height - start_value) / rise_days
    decline_rate = (peak_height - end_value) / fall_days
    gap = None if previous_end is None else int((start_date - previous_end).days)

    return {
        'country': country_df.iloc[0]['country'],
        'wave_id': wave_id,
        'start_date': start_date,
        'peak_date': peak_date,
        'end_date': end_date,
        'duration_days': duration_days,
        'peak_height': peak_height,
        'growth_rate_per_day': growth_rate,
        'decline_rate_per_day': decline_rate,
        'days_since_previous_wave': gap,
    }


def detect_waves(country_df: pd.DataFrame, threshold_per_100k=WAVE_THRESHOLD_PER_100K, min_wave_days=MIN_WAVE_DAYS) -> pd.DataFrame:
    country_df = country_df.sort_values('date').reset_index(drop=True).copy()
    above = country_df['cases_per_100k_7d'].fillna(0).ge(threshold_per_100k).to_numpy()

    waves = []
    start = None
    previous_end = None
    wave_id = 0

    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            end = i - 1
            if end - start + 1 >= min_wave_days:
                wave_id += 1
                wave = summarize_wave(country_df, start, end, wave_id, previous_end)
                waves.append(wave)
                previous_end = country_df.iloc[end]['date']
            start = None

    if start is not None:
        end = len(country_df) - 1
        if end - start + 1 >= min_wave_days:
            wave_id += 1
            waves.append(summarize_wave(country_df, start, end, wave_id, previous_end))

    return pd.DataFrame(waves)


def run_wave_detection(df: pd.DataFrame) -> pd.DataFrame:
    wave_tables = []
    for country in COUNTRIES:
        cdf = df[df['country'] == country].copy()
        waves = detect_waves(cdf)
        if not waves.empty:
            wave_tables.append(waves)
    return pd.concat(wave_tables, ignore_index=True) if wave_tables else pd.DataFrame()


def summarize_waves_by_country(wave_summary: pd.DataFrame) -> pd.DataFrame:
    if wave_summary.empty:
        return pd.DataFrame(columns=[
            'country', 'number_of_waves', 'average_peak_height', 'average_duration_days',
            'average_growth_rate', 'average_decline_rate'
        ])

    return (
        wave_summary.groupby('country')
        .agg(
            number_of_waves=('wave_id', 'count'),
            average_peak_height=('peak_height', 'mean'),
            average_duration_days=('duration_days', 'mean'),
            average_growth_rate=('growth_rate_per_day', 'mean'),
            average_decline_rate=('decline_rate_per_day', 'mean'),
        )
        .reset_index()
    )


def sensitivity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    sensitivity_rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        for country in COUNTRIES:
            cdf = df[df['country'] == country].copy()
            waves = detect_waves(cdf, threshold_per_100k=threshold, min_wave_days=MIN_WAVE_DAYS)
            if waves.empty:
                sensitivity_rows.append({
                    'country': country,
                    'threshold_per_100k': threshold,
                    'wave_count': 0,
                    'mean_peak_height': np.nan,
                    'mean_duration_days': np.nan,
                })
            else:
                sensitivity_rows.append({
                    'country': country,
                    'threshold_per_100k': threshold,
                    'wave_count': int(len(waves)),
                    'mean_peak_height': float(waves['peak_height'].mean()),
                    'mean_duration_days': float(waves['duration_days'].mean()),
                })
    return pd.DataFrame(sensitivity_rows)
