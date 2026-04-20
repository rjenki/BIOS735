import matplotlib.pyplot as plt
import pandas as pd

from .config import COUNTRIES, FIGURES_DIR, TABLES_DIR
from .io_utils import slugify

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

plt.rcParams['figure.figsize'] = (10, 5)


def save_table(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(TABLES_DIR / filename, index=False)


def plot_smoothed_time_series(df: pd.DataFrame) -> None:
    for country in COUNTRIES:
        cdf = df[df['country'] == country].copy()
        plt.figure()
        plt.plot(cdf['date'], cdf['cases_per_100k'], alpha=0.35, label='Daily cases per 100k')
        plt.plot(cdf['date'], cdf['cases_per_100k_7d'], linewidth=2, label='7-day rolling average')
        plt.title(f'COVID-19 Cases per 100k: {country}')
        plt.xlabel('Date')
        plt.ylabel('Cases per 100k')
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'smoothed_cases_{slugify(country)}.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_detected_waves(df: pd.DataFrame, wave_summary: pd.DataFrame) -> None:
    for country in COUNTRIES:
        cdf = df[df['country'] == country].copy()
        cwaves = wave_summary[wave_summary['country'] == country].copy()
        plt.figure()
        plt.plot(cdf['date'], cdf['cases_per_100k_7d'], linewidth=2, label='7-day rolling average')
        for _, row in cwaves.iterrows():
            plt.axvspan(row['start_date'], row['end_date'], alpha=0.2)
            plt.axvline(row['peak_date'], linestyle='--', linewidth=1)
        plt.title(f'Detected COVID-19 Waves: {country}')
        plt.xlabel('Date')
        plt.ylabel('Smoothed cases per 100k')
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'detected_waves_{slugify(country)}.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_forecasts(forecast_plots: dict) -> None:
    for country in COUNTRIES:
        plt.figure()
        actual_drawn = False
        for (c, model_name), plot_df in forecast_plots.items():
            if c != country:
                continue
            if not actual_drawn:
                plt.plot(plot_df['date'], plot_df['actual'], linewidth=2, label='Actual')
                actual_drawn = True
            plt.plot(plot_df['date'], plot_df['predicted'], label=model_name)
        plt.title(f'Forecast Comparison: {country}')
        plt.xlabel('Date')
        plt.ylabel('Cases per 100k (7-day average)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'forecast_comparison_{slugify(country)}.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_metric_bars(forecast_results: pd.DataFrame) -> None:
    if forecast_results.empty:
        return
    for metric in ['MAE', 'RMSE', 'sMAPE', 'train_seconds', 'predict_seconds']:
        pivot_df = forecast_results.pivot(index='country', columns='model', values=metric)
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot_df.plot(kind='bar', ax=ax)
        ax.set_title(f'Model Comparison by {metric}')
        ax.set_xlabel('Country')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=0)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f'model_comparison_{metric.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
