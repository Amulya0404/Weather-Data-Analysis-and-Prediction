# Weather Data Analysis and Prediction

Analyzes historical daily temperature data, measures long-term trends, and forecasts future temperatures using regression. A good introduction to time-series data and basic regression techniques.

## Files

```
.
├── weather_analysis.py    # main script
├── forecast.csv           # example output: daily forecast with 95% range
└── weather_report.png     # example output: three-panel plot
```

## Features

- Loads a CSV (any date column plus a daily temperature column) or uses built-in synthetic demo data
- **Cleaning:** sorts by date, removes duplicate dates, fills short gaps by interpolation
- **Analysis:** summary statistics, hottest and coldest day, average temperature by month, warming trend per decade
- **Model:** linear regression with a time trend plus yearly seasonality (sine and cosine terms)
- **Honest evaluation:** trains on the earlier 80% of the data and tests on the final 20% (no shuffling, so no future information leaks into training), and compares against two baselines using MAE, RMSE and R²
- **Forecast:** predicts the next N days with a 95% range and saves `forecast.csv`
- **Plots:** saves a three-panel figure, `weather_report.png`

## Install

```bash
pip install pandas numpy scikit-learn matplotlib
```

Requires Python 3.9+.

## Usage

```bash
python weather_analysis.py                       # demo with synthetic data
python weather_analysis.py --csv data.csv --date-col DATE --temp-col TAVG --days 365
```

| Option | Description | Default |
|---|---|---|
| `--csv` | Path to your CSV file | demo data |
| `--date-col` | Name of the date column | `date` |
| `--temp-col` | Name of the temperature column | `temp` |
| `--days` | Number of days to forecast | `365` |
| `--out` | Output plot filename | `weather_report.png` |

Free real-world data sources: [NOAA Climate Data Online](https://www.ncei.noaa.gov/cdo-web/), [Meteostat](https://meteostat.net/), or weather datasets on Kaggle.

## How it works

The model predicts temperature from the date alone:

```
temp = a + b·t + c1·sin(2π·doy/365.25) + c2·cos(2π·doy/365.25) [+ 2nd harmonic]
```

- `t` is time in years, which captures the long-term trend
- `doy` is the day of the year; the sine and cosine terms capture the yearly cycle
- Because the features come only from dates, the model can forecast any future period

## Example results (synthetic demo data)

**Summary**

```
Period: 2014-01-01 to 2023-12-29 (3650 days)
mean 12.37   std 8.33   min -6.54   max 32.22
Hottest day: 2022-08-08 (32.2)
Coldest day: 2022-01-14 (-6.5)
Long-term trend: +1.13 degrees per decade
```

**Model comparison** (held-out final 20%)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 7.38 | 8.45 | -0.005 |
| Baseline (day-of-year average) | 2.56 | 3.18 | 0.858 |
| Linear + 1 harmonic | 2.37 | 2.96 | 0.877 |
| Linear + 2 harmonics | 2.37 | 2.95 | 0.877 |

The fitted trend was +0.93 degrees per decade, against a true value of +0.8 in the synthetic data.

**Forecast (monthly averages, 2024)**

```
Jan 2.2   Feb 3.1   Mar 6.7   Apr 12.1   May 17.7   Jun 22.1
Jul 23.9  Aug 22.6  Sep 18.8  Oct 13.5   Nov 8.2    Dec 4.2
```

**Plot panels** (`weather_report.png`)
1. Ten years of daily temperatures with a 30-day average, the model fit, and the forecast with its 95% range
2. Annual mean temperatures with a trend line
3. Average temperature by month, with standard deviation error bars

## Limitations

- A model built only from date and season predicts the typical temperature for a day, not tomorrow's actual weather. Day-to-day variation is mostly noise, so R² tops out near 0.88 on this data.
- The 95% range covers individual days and does not include uncertainty in the trend, so long-range forecasts are less certain than the band suggests.
- The example numbers come from synthetic data. Results on real data will differ.

## Ideas to extend

- Add lag features (yesterday's temperature) for short-term forecasts
- Try `RandomForestRegressor`, SARIMA (`statsmodels`) or Prophet
- Add more variables such as humidity and pressure
- Use `TimeSeriesSplit` for cross-validation
