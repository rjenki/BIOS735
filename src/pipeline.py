from .data_processing import load_raw_data, preprocess_data, summarize_coverage
from .forecasting import run_forecasting
from .io_utils import ensure_directories
from .reporting import (
    plot_detected_waves,
    plot_forecasts,
    plot_metric_bars,
    plot_smoothed_time_series,
    save_table,
)
from .wave_analysis import run_wave_detection, sensitivity_analysis, summarize_waves_by_country


def run_pipeline():
    ensure_directories()

    raw_df = load_raw_data()
    df = preprocess_data(raw_df)
    coverage = summarize_coverage(df)

    wave_summary = run_wave_detection(df)
    wave_country_summary = summarize_waves_by_country(wave_summary)
    sensitivity_df = sensitivity_analysis(df)

    forecast_results, best_models, forecast_plots = run_forecasting(df)

    save_table(df, 'preprocessed_data.csv')
    save_table(coverage, 'coverage_summary.csv')
    save_table(wave_summary, 'wave_summary.csv')
    save_table(wave_country_summary, 'wave_country_summary.csv')
    save_table(sensitivity_df, 'wave_sensitivity_analysis.csv')
    save_table(forecast_results, 'forecast_results.csv')
    save_table(best_models, 'best_models_by_country.csv')

    plot_smoothed_time_series(df)
    plot_detected_waves(df, wave_summary)
    plot_forecasts(forecast_plots)
    plot_metric_bars(forecast_results)

    return {
        'preprocessed_data': df,
        'coverage_summary': coverage,
        'wave_summary': wave_summary,
        'wave_country_summary': wave_country_summary,
        'sensitivity_df': sensitivity_df,
        'forecast_results': forecast_results,
        'best_models': best_models,
    }
