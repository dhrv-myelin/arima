"""
arima_cylinder.py
=================
ARIMA train/test wrapper for cylinder stroke time series.

DATASET QUICK-REFERENCE
-----------------------
Source CSV: cylinder_strokes.csv
Each raw row = one cylinder actuation, with duration_ms derived from
timestamp-differencing of machine log events:
    Extension : valve[ON]  → [WORK Complete]
    Retraction: valve[OFF] → [HOME Complete]

THE 8 TIME SERIES  (4 cylinders × 2 directions)
------------------------------------------------
  cylinder        direction   mean_ms   character
  Lifting         extend      ~147 ms   flat / stationary
  Lifting         retract     ~124 ms   flat / stationary
  Middle          extend      ~169 ms   ★ DRIFTS +10% over 35 days  ← best ARIMA target
  Middle          retract     ~124 ms   flat / stationary
  Front_barrier   extend      ~135 ms   noisy / volatile (high stdev)
  Front_barrier   retract     ~ 96 ms   flat, fastest cylinder
  Rear_barrier    extend      ~132 ms   flat / stationary
  Rear_barrier    retract     ~208 ms   flat, slowest cylinder

WHAT YOU MODEL
--------------
We aggregate raw strokes → one row per (date, cylinder, direction).
Each series has 35 daily data points (Feb 2 – Mar 9).
Four metrics are available per series:

  mean_ms   daily mean stroke duration          → primary trend signal
  std_ms    daily within-day spread             → variability signal
  p95_ms    daily 95th-percentile               → slow-stroke severity
  slow_pct  daily % of strokes > 500 ms         → anomaly rate (~5.3%, stationary)

Total: 8 series × 4 metrics = 32 selectable time series.

USAGE
-----
    from arima_cylinder import run_arima, list_series

    r = run_arima("Middle", "extend", metric="mean_ms", test_size=7)
    r = run_arima("Lifting", "extend", metric="slow_pct", test_size=7)

    r["model"].summary()
    r["forecast"]
    r["mae"], r["rmse"], r["mape"]
    r["order"]

    list_series()
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")

CSV_PATH = "./datasets/cylinder_strokes.csv"
SLOW_THRESH = 500

CYLINDERS = ["Lifting", "Middle", "Front_barrier", "Rear_barrier"]
DIRECTIONS = ["extend", "retract"]
METRICS = ["mean_ms", "std_ms", "p95_ms", "slow_pct"]


def load_daily(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load CSV and return daily aggregates (280 rows: 35 days x 8 series)."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    daily = (
        df.groupby(["date", "cylinder", "direction"])["duration_ms"]
        .agg(
            mean_ms="mean",
            std_ms="std",
            p95_ms=lambda x: x.quantile(0.95),
            slow_pct=lambda x: (x > SLOW_THRESH).mean() * 100,
        )
        .reset_index()
    )
    return daily


def extract_series(daily, cylinder, direction, metric="mean_ms") -> pd.Series:
    """Pull one time series out of the daily table as a date-indexed pd.Series."""
    for val, choices in [
        (cylinder, CYLINDERS),
        (direction, DIRECTIONS),
        (metric, METRICS),
    ]:
        if val not in choices:
            raise ValueError(f"'{val}' not valid. Choose from {choices}")
    mask = (daily["cylinder"] == cylinder) & (daily["direction"] == direction)
    s = daily.loc[mask, ["date", metric]].set_index("date")[metric].sort_index()
    s.name = f"{cylinder}__{direction}__{metric}"
    return s


def run_arima(
    cylinder: str,
    direction: str,
    metric: str = "mean_ms",
    test_size: int = 7,
    csv_path: str = CSV_PATH,
    seasonal: bool = False,
    m: int = 7,
    plot: bool = True,
    verbose: bool = True,
) -> dict:
    """
    ARIMA train/test pipeline for any cylinder stroke time series.

    Parameters
    ----------
    cylinder   "Lifting" | "Middle" | "Front_barrier" | "Rear_barrier"
    direction  "extend" | "retract"
    metric     "mean_ms" | "std_ms" | "p95_ms" | "slow_pct"
    test_size  days to hold out (default 7)
    csv_path   path to cylinder_strokes.csv
    seasonal   whether to fit SARIMA seasonal component
    m          seasonal period if seasonal=True (7 = weekly)
    plot       draw diagnostic chart
    verbose    print auto_arima search log + summary

    Returns dict with keys:
        series, train, test, model, forecast, conf_int,
        mae, rmse, mape, order, seasonal_order
    """
    # 1. Load & extract
    daily = load_daily(csv_path)
    series = extract_series(daily, cylinder, direction, metric)
    unit = "%" if metric == "slow_pct" else "ms"

    if len(series) < test_size + 5:
        raise ValueError(
            f"Series too short ({len(series)} pts) for test_size={test_size}"
        )

    # 2. Train / test split
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Series : {series.name}")
        print(
            f"  Train  : {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} days)"
        )
        print(
            f"  Test   : {test.index[0].date()} → {test.index[-1].date()}  ({len(test)} days)"
        )
        print(f"  Range  : {series.min():.1f} – {series.max():.1f} {unit}")
        print(f"{'='*60}\n")

    # 3. Fit — auto_arima searches (p,d,q) space and picks lowest AIC
    model = auto_arima(
        train,
        seasonal=seasonal,
        m=m if seasonal else 1,
        stepwise=True,
        information_criterion="aic",
        error_action="ignore",
        suppress_warnings=True,
        trace=verbose,
    )

    if verbose:
        print("\n" + str(model.summary()))

    # 4. Forecast
    # NOTE: pmdarima.predict() returns an integer-indexed pd.Series when the
    # training index has no recognised datetime frequency. We extract .values
    # first, then re-index to the test dates so metrics work cleanly.
    raw_fc, conf_int = model.predict(n_periods=test_size, return_conf_int=True)
    fc_arr = raw_fc.values if hasattr(raw_fc, "values") else np.asarray(raw_fc)
    forecast = pd.Series(fc_arr, index=test.index, name="forecast")

    # 5. Metrics
    mae = mean_absolute_error(test.values, forecast.values)
    rmse = np.sqrt(mean_squared_error(test.values, forecast.values))
    mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100

    if verbose:
        print(f"\n── Forecast metrics ───────────────────────────")
        print(
            f"  Order  : ARIMA{model.order}"
            + (f" x SARIMA{model.seasonal_order}" if seasonal else "")
        )
        print(f"  MAE    : {mae:.3f} {unit}")
        print(f"  RMSE   : {rmse:.3f} {unit}")
        print(f"  MAPE   : {mape:.2f}%")

    # 6. Plot
    if plot:
        _plot(series, train, test, forecast, conf_int, model, mae, rmse, mape)

    return {
        "series": series,
        "train": train,
        "test": test,
        "model": model,
        "forecast": forecast,
        "conf_int": conf_int,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "order": model.order,
        "seasonal_order": model.seasonal_order if seasonal else None,
    }


def _plot(series, train, test, forecast, conf_int, model, mae, rmse, mape):
    unit = "%" if "slow_pct" in series.name else "ms"
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"ARIMA{model.order}  ·  {series.name}",
        fontsize=13,
        y=0.99,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    # Panel 1 — full series + forecast
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(train.index, train.values, color="#378ADD", lw=1.5, label="Train")
    ax1.plot(test.index, test.values, color="#1D9E75", lw=1.5, label="Test (actual)")
    ax1.plot(
        forecast.index, forecast.values, "--", color="#D85A30", lw=2, label="Forecast"
    )
    ax1.fill_between(
        forecast.index,
        conf_int[:, 0],
        conf_int[:, 1],
        color="#D85A30",
        alpha=0.15,
        label="95% CI",
    )
    ax1.axvline(test.index[0], color="gray", lw=0.8, linestyle=":")
    ax1.set_ylabel(unit)
    ax1.set_title("Full series — train / test / forecast", fontsize=11)
    ax1.legend(fontsize=9, ncol=4)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2 — test window zoom
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(
        test.index, test.values, "o-", color="#1D9E75", lw=1.5, ms=5, label="Actual"
    )
    ax2.plot(
        test.index,
        forecast.values,
        "s--",
        color="#D85A30",
        lw=1.5,
        ms=5,
        label="Forecast",
    )
    ax2.fill_between(
        forecast.index, conf_int[:, 0], conf_int[:, 1], color="#D85A30", alpha=0.15
    )
    ax2.set_title("Forecast vs actual (test window)", fontsize=11)
    ax2.set_ylabel(unit)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.text(
        0.03,
        0.05,
        f"MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%",
        transform=ax2.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    # Panel 3 — residuals over time
    residuals = pd.Series(model.resid(), index=train.index)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(residuals.index, residuals.values, color="#534AB7", lw=1)
    ax3.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax3.fill_between(
        residuals.index,
        residuals.values,
        0,
        where=residuals.values > 0,
        color="#534AB7",
        alpha=0.15,
    )
    ax3.fill_between(
        residuals.index,
        residuals.values,
        0,
        where=residuals.values < 0,
        color="#D85A30",
        alpha=0.15,
    )
    ax3.set_title("Training residuals", fontsize=11)
    ax3.set_ylabel(unit)
    ax3.grid(axis="y", alpha=0.3)

    # Panel 4 — residual histogram
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(
        residuals.values, bins=15, color="#534AB7", alpha=0.7, edgecolor="white", lw=0.4
    )
    ax4.axvline(0, color="gray", lw=1, linestyle="--")
    ax4.set_title("Residual distribution", fontsize=11)
    ax4.set_xlabel(f"Residual ({unit})")
    ax4.set_ylabel("Count")
    ax4.grid(axis="y", alpha=0.3)

    # Panel 5 — residual ACF
    ax5 = fig.add_subplot(gs[2, 1])
    max_lags = min(15, len(residuals) // 2 - 1)
    plot_acf(residuals.values, lags=max_lags, ax=ax5, color="#534AB7")
    ax5.set_title("Residual ACF (should look like white noise)", fontsize=11)
    ax5.grid(axis="y", alpha=0.3)

    plt.savefig("arima_result.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("  Plot saved → arima_result.png")


def list_series():
    """Print all 32 available (cylinder, direction, metric) combinations."""
    desc = {
        "mean_ms": "mean stroke duration per day",
        "std_ms": "within-day variability",
        "p95_ms": "95th-percentile duration",
        "slow_pct": "% of strokes >500ms (anomaly rate, stationary)",
    }
    print(f"\n{'#':<4} {'cylinder':<16} {'direction':<10} {'metric':<12}  description")
    print("-" * 68)
    i = 1
    for cyl in CYLINDERS:
        for dirn in DIRECTIONS:
            for metric in METRICS:
                print(f"{i:<4} {cyl:<16} {dirn:<10} {metric:<12}  {desc[metric]}")
                i += 1
    print(f"\nTotal: {i-1} series")


if __name__ == "__main__":
    # Drifting series — best ARIMA target
    result = run_arima("Middle", "extend", metric="mean_ms", test_size=7)

    # Anomaly rate — stationary, no differencing needed
    # result = run_arima("Lifting", "extend", metric="slow_pct", test_size=7)

    # Slowest cylinder
    # result = run_arima("Rear_barrier", "retract", metric="mean_ms", test_size=7)

    list_series()

