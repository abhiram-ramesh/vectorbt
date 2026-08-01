"""
Trendline Slope Strategy using vectorbt.

Fits rolling support/resistance trendlines to OHLC data, then trades
based on slope crossovers:
  - BUY when support slope crosses above a threshold (uptrend forming)
  - SELL when resistance slope crosses below a threshold (downtrend forming)

Parameters (tunable via autoresearch):
  LOOKBACK        — rolling window for trendline fitting
  SUPPORT_THRESH  — slope threshold for entry signal
  RESIST_THRESH   — slope threshold for exit signal
  FEES            — trading fees as a fraction
  INIT_CASH       — starting capital
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

# ============================================================
# Strategy Parameters (autoresearch tunes these)
# ============================================================
LOOKBACK = 30
SUPPORT_THRESH = 0.0
RESIST_THRESH = 0.0
FEES = 0.001
INIT_CASH = 10_000

# ============================================================
# Trendline fitting (from user's code, cleaned up)
# ============================================================

def check_trend_line(support: bool, pivot: int, slope: float, y: np.ndarray) -> float:
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y

    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    return (diffs ** 2.0).sum()


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.ndarray):
    slope_unit = (y.max() - y.min()) / len(y)
    min_step = 0.0001
    curr_step = 1.0

    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert best_err >= 0.0

    get_derivative = True
    derivative = None

    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:
                break  # derivative failed, stop optimizing
            get_derivative = False

        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True

    return best_slope, -best_slope * pivot + y[pivot]


def compute_trendline_slopes(high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int):
    """Compute rolling support/resistance slopes over the full series."""
    n = len(close)
    support_slopes = np.full(n, np.nan)
    resist_slopes = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        h = high[i - lookback + 1: i + 1]
        l = low[i - lookback + 1: i + 1]
        c = close[i - lookback + 1: i + 1]

        x = np.arange(lookback)
        coefs = np.polyfit(x, c, 1)
        line_points = coefs[0] * x + coefs[1]

        upper_pivot = (h - line_points).argmax()
        lower_pivot = (l - line_points).argmin()

        sup_slope, _ = optimize_slope(True, lower_pivot, coefs[0], l)
        res_slope, _ = optimize_slope(False, upper_pivot, coefs[0], h)

        support_slopes[i] = sup_slope
        resist_slopes[i] = res_slope

    return support_slopes, resist_slopes


# ============================================================
# Generate synthetic BTC-like data for demo
# ============================================================
np.random.seed(42)
n_days = 500
returns = np.random.normal(0.0003, 0.025, n_days)
base_price = 30000 * np.cumprod(1 + returns)

# Synthesize OHLC from close
close = base_price
high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
open_ = close * (1 + np.random.normal(0, 0.005, n_days))

dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
ohlc = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=dates)

# ============================================================
# Compute trendline slopes on log prices
# ============================================================
log_high = np.log(ohlc["High"].values)
log_low = np.log(ohlc["Low"].values)
log_close = np.log(ohlc["Close"].values)

support_slopes, resist_slopes = compute_trendline_slopes(log_high, log_low, log_close, LOOKBACK)

ohlc["support_slope"] = support_slopes
ohlc["resist_slope"] = resist_slopes

# ============================================================
# Generate entry/exit signals from slopes
# ============================================================
support = pd.Series(support_slopes, index=dates)
resist = pd.Series(resist_slopes, index=dates)

# Buy: support slope crosses above threshold (uptrend starting)
entries = (support > SUPPORT_THRESH) & (support.shift(1) <= SUPPORT_THRESH)
# Sell: resistance slope crosses below threshold (downtrend starting)
exits = (resist < RESIST_THRESH) & (resist.shift(1) >= RESIST_THRESH)

# ============================================================
# Backtest with vectorbt
# ============================================================
pf = vbt.Portfolio.from_signals(
    ohlc["Close"],
    entries=entries,
    exits=exits,
    init_cash=INIT_CASH,
    fees=FEES,
)

# ============================================================
# Results
# ============================================================
bh_return = (ohlc["Close"].iloc[-1] / ohlc["Close"].iloc[0]) - 1

print("=== Trendline Slope Strategy ===")
print(f"Lookback:         {LOOKBACK}")
print(f"Support thresh:   {SUPPORT_THRESH}")
print(f"Resist thresh:    {RESIST_THRESH}")
print()
print(f"Total Return:     {pf.total_return():.2%}")
print(f"Sharpe Ratio:     {pf.sharpe_ratio():.4f}")
print(f"Max Drawdown:     {pf.max_drawdown():.2%}")
print(f"Total Trades:     {pf.trades.count()}")
print(f"Win Rate:         {pf.trades.win_rate():.2%}")
print(f"Profit Factor:    {pf.trades.profit_factor():.4f}")
print()
print(f"Buy & Hold:       {bh_return:.2%}")
print(f"Alpha:            {pf.total_return() - bh_return:.2%}")
print()
print("=== Trades ===")
print(pf.trades.records_readable.to_string())
