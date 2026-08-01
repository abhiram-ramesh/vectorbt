"""
Parameter sweep for the trendline strategy using vectorbt.

Tests multiple lookback windows and slope thresholds, then prints
a ranked table of the best parameter combinations by Sharpe ratio.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from trendline_strategy import compute_trendline_slopes

# ============================================================
# Generate data
# ============================================================
np.random.seed(42)
n_days = 500
returns = np.random.normal(0.0003, 0.025, n_days)
base_price = 30000 * np.cumprod(1 + returns)
close = base_price
high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
ohlc = pd.DataFrame({"High": high, "Low": low, "Close": close}, index=dates)

# ============================================================
# Sweep parameters
# ============================================================
lookbacks = [15, 20, 30, 45, 60]
thresholds = [-0.002, -0.001, 0.0, 0.001, 0.002]

results = []
t0 = time.time()

for lb in lookbacks:
    log_h = np.log(ohlc["High"].values)
    log_l = np.log(ohlc["Low"].values)
    log_c = np.log(ohlc["Close"].values)
    sup_slopes, res_slopes = compute_trendline_slopes(log_h, log_l, log_c, lb)

    support = pd.Series(sup_slopes, index=dates)
    resist = pd.Series(res_slopes, index=dates)

    for sup_t in thresholds:
        for res_t in thresholds:
            entries = (support > sup_t) & (support.shift(1) <= sup_t)
            exits = (resist < res_t) & (resist.shift(1) >= res_t)

            pf = vbt.Portfolio.from_signals(
                ohlc["Close"], entries=entries, exits=exits,
                init_cash=10_000, fees=0.001,
            )

            n_trades = pf.trades.count()
            if n_trades < 2:
                continue

            results.append({
                "lookback": lb,
                "sup_thresh": sup_t,
                "res_thresh": res_t,
                "sharpe": pf.sharpe_ratio(),
                "return": pf.total_return(),
                "max_dd": pf.max_drawdown(),
                "trades": n_trades,
                "win_rate": pf.trades.win_rate(),
            })

elapsed = time.time() - t0

df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
print(f"Swept {len(lookbacks)} lookbacks x {len(thresholds)}^2 thresholds = {len(lookbacks)*len(thresholds)**2} combos in {elapsed:.1f}s")
print(f"Valid combos (>=2 trades): {len(df)}")
print()
print("=== Top 10 by Sharpe Ratio ===")
print(df.head(10).to_string(index=False, float_format="%.4f"))
print()
print("=== Bottom 5 ===")
print(df.tail(5).to_string(index=False, float_format="%.4f"))
