#!/usr/bin/env python3
"""
Weather Data Analysis and Temperature Prediction

Analyzes historical daily temperature data and forecasts future trends using
regression with a long-term trend plus yearly seasonality.

Model:  temp = a + b*t + c1*sin(2*pi*doy/365.25) + c2*cos(2*pi*doy/365.25)
        (linear regression; optional 2nd harmonic and lag features are included
         in the "advanced" model for comparison)

Usage:
    python weather_analysis.py                       # uses synthetic demo data
    python weather_analysis.py --csv data.csv        # your own data
    python weather_analysis.py --csv data.csv --date-col DATE --temp-col TAVG --days 365

CSV needs one date column and one daily temperature column (e.g., from NOAA
Climate Data Online, Meteostat, or Kaggle weather datasets).
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # save plots to file; works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def make_demo_data(years: int = 10, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily temps (°C): seasonal cycle + slight warming trend + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-01", periods=years * 365, freq="D")
    doy = dates.dayofyear.values
    t_years = np.arange(len(dates)) / 365.25
    temp = (
        12
        + 11 * np.sin(2 * np.pi * (doy - 110) / 365.25)  # seasonal cycle
        + 0.08 * t_years                                  # ~0.8°C/decade warming
        + rng.normal(0, 3, len(dates))                    # daily noise
    )
    return pd.DataFrame({"date": dates, "temp": temp})


def load_csv(path: str, date_col: str, temp_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns or temp_col not in df.columns:
        raise SystemExit(f"Columns not found. Available columns: {list(df.columns)}")
    df = df[[date_col, temp_col]].rename(columns={date_col: "date", temp_col: "temp"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, drop duplicates/bad rows, fill small gaps by interpolation."""
    df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    df = df.set_index("date").asfreq("D")
    missing = int(df["temp"].isna().sum())
    if missing:
        print(f"Filling {missing} missing days by linear interpolation")
        df["temp"] = df["temp"].interpolate(limit=7).bfill().ffill()
    return df.reset_index()


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def describe(df: pd.DataFrame) -> None:
    print("\n=== Summary ===")
    print(f"Period: {df['date'].min().date()} to {df['date'].max().date()} "
          f"({len(df)} days)")
    print(df["temp"].describe().round(2).to_string())

    hottest = df.loc[df["temp"].idxmax()]
    coldest = df.loc[df["temp"].idxmin()]
    print(f"\nHottest day: {hottest['date'].date()} ({hottest['temp']:.1f})")
    print(f"Coldest day: {coldest['date'].date()} ({coldest['temp']:.1f})")

    monthly = df.groupby(df["date"].dt.month)["temp"].mean().round(1)
    print("\nAverage temperature by month:")
    print(monthly.to_string())

    # Long-term trend: regress annual means on year (skip partial years)
    yearly = df.groupby(df["date"].dt.year)["temp"].agg(["mean", "count"])
    yearly = yearly[yearly["count"] >= 350]
    if len(yearly) >= 3:
        slope, intercept = np.polyfit(yearly.index, yearly["mean"], 1)
        print(f"\nLong-term trend: {slope * 10:+.2f} degrees per decade "
              f"(from {len(yearly)} full years)")


# ----------------------------------------------------------------------------
# Modeling
# ----------------------------------------------------------------------------
def features(dates: pd.Series, start: pd.Timestamp, harmonics: int = 1) -> pd.DataFrame:
    """Trend + seasonal (Fourier) features from dates only, so we can forecast."""
    t = (dates - start).dt.days.values / 365.25
    doy = dates.dt.dayofyear.values
    X = {"trend": t}
    for k in range(1, harmonics + 1):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return pd.DataFrame(X)


def evaluate(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{label:<28} MAE={mae:5.2f}  RMSE={rmse:5.2f}  R²={r2:5.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_evaluate(df: pd.DataFrame, test_frac: float = 0.2):
    """Time-ordered split (never shuffle time series!) and compare models."""
    start = df["date"].min()
    split = int(len(df) * (1 - test_frac))
    train, test = df.iloc[:split], df.iloc[split:]

    print("\n=== Model comparison (held-out final "
          f"{int(test_frac * 100)}% of the data) ===")

    # Baseline: predict the training mean
    evaluate(test["temp"], np.full(len(test), train["temp"].mean()), "Baseline (mean)")

    # Baseline: seasonal average by day-of-year from training data
    doy_mean = train.groupby(train["date"].dt.dayofyear)["temp"].mean()
    clim = test["date"].dt.dayofyear.map(doy_mean).fillna(train["temp"].mean())
    evaluate(test["temp"], clim, "Baseline (climatology)")

    # Regression models
    results = {}
    for name, h in [("Linear + 1 harmonic", 1), ("Linear + 2 harmonics", 2)]:
        model = LinearRegression().fit(features(train["date"], start, h), train["temp"])
        pred = model.predict(features(test["date"], start, h))
        results[name] = (model, h, evaluate(test["temp"], pred, name))

    best = min(results, key=lambda k: results[k][2]["rmse"])
    print(f"\nBest model: {best}")
    model, h, _ = results[best]

    trend_coef = model.coef_[0]
    print(f"Fitted warming trend: {trend_coef * 10:+.2f} degrees per decade")

    # Refit on all data for the final forecast
    final = LinearRegression().fit(features(df["date"], start, h), df["temp"])
    residual_std = np.std(df["temp"] - final.predict(features(df["date"], start, h)))
    return final, h, start, residual_std, (train, test)


def forecast(df, model, h, start, resid_std, days: int) -> pd.DataFrame:
    future_dates = pd.Series(
        pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=days, freq="D")
    )
    pred = model.predict(features(future_dates, start, h))
    return pd.DataFrame({
        "date": future_dates,
        "predicted_temp": pred,
        "lower_95": pred - 1.96 * resid_std,
        "upper_95": pred + 1.96 * resid_std,
    })


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def make_plots(df, model, h, start, fc, out="weather_report.png"):
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))

    # 1) History + fit + forecast
    ax = axes[0]
    ax.plot(df["date"], df["temp"], color="lightgray", lw=0.6, label="Daily observed")
    ax.plot(df["date"], df["temp"].rolling(30, center=True).mean(),
            color="tab:blue", lw=1.2, label="30-day average")
    ax.plot(df["date"], model.predict(features(df["date"], start, h)),
            color="tab:red", lw=1, alpha=0.8, label="Model fit")
    ax.plot(fc["date"], fc["predicted_temp"], color="tab:green", lw=1.5, label="Forecast")
    ax.fill_between(fc["date"], fc["lower_95"], fc["upper_95"],
                    color="tab:green", alpha=0.15, label="95% range (daily)")
    ax.set_title("Historical temperature, model fit and forecast")
    ax.set_ylabel("Temperature")
    ax.legend(loc="upper left", ncol=3, fontsize=8)

    # 2) Annual means + trend
    ax = axes[1]
    yearly = df.groupby(df["date"].dt.year)["temp"].agg(["mean", "count"])
    yearly = yearly[yearly["count"] >= 350]
    ax.bar(yearly.index, yearly["mean"], color="tab:orange", alpha=0.7)
    if len(yearly) >= 3:
        slope, intercept = np.polyfit(yearly.index, yearly["mean"], 1)
        ax.plot(yearly.index, slope * yearly.index + intercept, "k--",
                label=f"Trend: {slope * 10:+.2f} per decade")
        ax.legend()
    lo, hi = yearly["mean"].min(), yearly["mean"].max()
    ax.set_ylim(lo - (hi - lo) * 0.5, hi + (hi - lo) * 0.3)
    ax.set_title("Annual mean temperature")
    ax.set_ylabel("Temperature")

    # 3) Monthly climatology
    ax = axes[2]
    monthly = df.groupby(df["date"].dt.month)["temp"]
    ax.errorbar(range(1, 13), monthly.mean(), yerr=monthly.std(), fmt="o-",
                capsize=3, color="tab:purple")
    ax.set_xticks(range(1, 13))
    ax.set_title("Average temperature by month (± 1 std dev)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nSaved plots to {out}")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Weather analysis and temperature forecast")
    p.add_argument("--csv", help="Path to CSV file (omit to use synthetic demo data)")
    p.add_argument("--date-col", default="date", help="Date column name")
    p.add_argument("--temp-col", default="temp", help="Temperature column name")
    p.add_argument("--days", type=int, default=365, help="Days to forecast (default 365)")
    p.add_argument("--out", default="weather_report.png", help="Output plot filename")
    args = p.parse_args()

    if args.csv:
        df = load_csv(args.csv, args.date_col, args.temp_col)
    else:
        print("No CSV given - using synthetic demo data.")
        df = make_demo_data()

    df = clean(df)
    describe(df)
    model, h, start, resid_std, _ = train_and_evaluate(df)
    fc = forecast(df, model, h, start, resid_std, args.days)

    print("\n=== Forecast: monthly averages ===")
    monthly_fc = fc.groupby(fc["date"].dt.to_period("M"))["predicted_temp"].mean().round(1)
    print(monthly_fc.to_string())

    fc.to_csv("forecast.csv", index=False)
    print("\nSaved daily forecast to forecast.csv")
    make_plots(df, model, h, start, fc, args.out)


if __name__ == "__main__":
    main()
