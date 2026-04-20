# COVID-19 Wave Characterization and Forecasting

This project analyzes COVID-19 case trends across four countries (Brazil, Germany, Japan, and the United States) using publicly available data from Our World in Data. First, the data are pre-processed and smoothed to identify distinct epidemic waves and summarize their key characteristics, such as duration and peak intensity. Next, several forecasting models are applied, including simple baselines (naive and moving average), a classical time series model (ARIMA), and a machine learning approach (random forest). These models are compared based on prediction accuracy and computational efficiency. The project aims to show how different modeling strategies perform under varying epidemic conditions and to provide practical insights for short-term forecasting of infectious disease data.

## Structure
`run_analysis.py`: Runs the full project in order.  
`src/covid_project/config.py`: Project settings. Edit to change countries and wave definition settings.  
`src/covid_project/data_processing.py`: Loading, filtering, trimming, and feature preparation.  
`src/covid_project/wave_analysis.py`: Wave detection and sensitivity analysis.  
`src/covid_project/forecasting.py`: Forecasting models and evaluation.  
`src/covid_project/reporting.py`: Saves figures and CSV tables.  
`src/covid_project/pipeline.py`: Organizes the full workflow.  
`data/compact.csv`: Our World in Data source data set.  
`outputs/figures/`: Exported PNG figures.  
`outputs/tables/`: Exported CSV tables.  

## Running The Program
First, ensure all required packages are installed. 
```bash
python -m pip install -r requirements.txt
```

Then run:
```bash
python run_analysis.py
```
