"""
Production Strategy: S/R + RSI Filter + Hawkes Volatility (L+S)

The validated BTC daily strategy from our research.
MCPT p-value < 0.01, OOS Sharpe 1.05-1.29, walk-forward 7/8 positive periods.

Long entries: S/R level break (RSI>50) + Hawkes vol explosion (upward)
Long exits:  S/R level break (downward)
Short entries: S/R level break (RSI<50) + Hawkes vol explosion (downward)
Short exits: S/R level break (upward)

Usage:
    python sr_strategy.py                    # Run on default BTC data
    python sr_strategy.py /path/to/data.csv  # Custom data
"""

import sys
import os
import math
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vectorbt as vbt


# ============================================================
# Pure numpy/pandas replacements (no scipy/pandas_ta needed)
# ============================================================

def _gaussian_kde(data, bw_factor, weights, grid):
    """Gaussian KDE matching scipy.stats.gaussian_kde behaviour."""
    norm_weights = weights / weights.sum()
    wmean = np.dot(data, norm_weights)
    wvar = np.dot(norm_weights, (data - wmean) ** 2)
    n_eff = 1.0 / np.dot(norm_weights, norm_weights)
    wvar = wvar * n_eff / (n_eff - 1.0)
    bw = bw_factor * math.sqrt(wvar)
    if bw <= 0:
        return np.zeros_like(grid)
    inv_bw = 1.0 / bw
    coeff = inv_bw / math.sqrt(2.0 * math.pi)
    diff = (grid[:, None] - data[None, :]) * inv_bw
    log_gauss = np.clip(-0.5 * diff * diff, -500, 0)
    gaussians = coeff * np.exp(log_gauss)
    return gaussians @ norm_weights


def _find_peaks(x, prominence_min):
    """Find local maxima with prominence filtering."""
    peaks = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            peaks.append(i)
    if not peaks:
        return np.array([], dtype=int), {}
    filtered = []
    for p in peaks:
        left_min = x[p]
        for j in range(p - 1, -1, -1):
            left_min = min(left_min, x[j])
            if x[j] > x[p]:
                break
        right_min = x[p]
        for j in range(p + 1, len(x)):
            right_min = min(right_min, x[j])
            if x[j] > x[p]:
                break
        prom = x[p] - max(left_min, right_min)
        if prom >= prominence_min:
            filtered.append(p)
    return np.array(filtered, dtype=int), {}


def _rsi(close, period=14):
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high, low, close, period=14):
    """Wilder ATR."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


# ============================================================
# S/R Levels via KDE Market Profile
# ============================================================

def find_levels(price, atr):
    weights = np.ones(len(price))
    min_v, max_v = np.min(price), np.max(price)
    step = (max_v - min_v) / 200
    if step <= 0:
        return []
    pr = np.arange(min_v, max_v, step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pdf = _gaussian_kde(price, atr * 3.0, weights, pr)
    peaks, _ = _find_peaks(pdf, np.max(pdf) * 0.25)
    return [np.exp(pr[p]) for p in peaks]


def compute_sr_signal(ohlc):
    """Compute S/R level penetration signal with RSI filter."""
    close_arr = ohlc['Close'].values
    log_c = np.log(close_arr)

    atr_sr = _atr(np.log(ohlc['High']), np.log(ohlc['Low']),
                  np.log(ohlc['Close']), 180)

    all_levels = [None] * len(ohlc)
    for i in range(180, len(ohlc)):
        a = atr_sr.iloc[i]
        if pd.isna(a) or a <= 0:
            continue
        try:
            all_levels[i] = find_levels(log_c[i - 179:i + 1], a)
        except:
            continue

    sr_signal = np.zeros(len(ohlc))
    curr_sig = 0.0
    for i in range(1, len(ohlc)):
        if all_levels[i] is not None:
            for level in all_levels[i]:
                if close_arr[i] > level and close_arr[i - 1] <= level:
                    curr_sig = 1.0
                elif close_arr[i] < level and close_arr[i - 1] >= level:
                    curr_sig = -1.0
        sr_signal[i] = curr_sig

    # RSI filter
    rsi = _rsi(ohlc['Close'], 14).values
    filtered = sr_signal.copy()
    for i in range(1, len(filtered)):
        if filtered[i] != filtered[i - 1]:
            if filtered[i] == 1.0 and (np.isnan(rsi[i]) or rsi[i] < 50):
                filtered[i] = filtered[i - 1]
            elif filtered[i] == -1.0 and (np.isnan(rsi[i]) or rsi[i] > 50):
                filtered[i] = filtered[i - 1]

    return filtered


# ============================================================
# Hawkes Volatility Process
# ============================================================

def compute_hawkes_signal(ohlc, kappa=0.25, quantile_lb=60):
    """Detect volatility compression -> explosion transitions."""
    log_h = np.log(ohlc['High'].values)
    log_l = np.log(ohlc['Low'].values)
    log_c = np.log(ohlc['Close'].values)
    close_arr = ohlc['Close'].values

    atr_180 = _atr(pd.Series(log_h), pd.Series(log_l),
                   pd.Series(log_c), 180).values
    norm_range = (log_h - log_l) / atr_180
    norm_range = np.where(np.isnan(norm_range) | np.isinf(norm_range),
                          0, norm_range)

    # Hawkes process
    alpha = np.exp(-kappa)
    hawkes = np.full(len(ohlc), np.nan)
    for i in range(1, len(ohlc)):
        if np.isnan(hawkes[i - 1]):
            hawkes[i] = norm_range[i]
        else:
            hawkes[i] = hawkes[i - 1] * alpha + norm_range[i]
    hawkes *= kappa

    # Quantile bands
    q05 = pd.Series(hawkes).rolling(quantile_lb).quantile(0.05).values
    q95 = pd.Series(hawkes).rolling(quantile_lb).quantile(0.95).values

    # Signal: compression -> explosion
    long_entry = np.zeros(len(ohlc), dtype=bool)
    short_entry = np.zeros(len(ohlc), dtype=bool)
    last_below = -1

    for i in range(1, len(ohlc)):
        if np.isnan(q05[i]) or np.isnan(q95[i]):
            continue
        if hawkes[i] < q05[i]:
            last_below = i
        if (hawkes[i] > q95[i] and hawkes[i - 1] <= q95[i - 1]
                and last_below > 0):
            if close_arr[i] > close_arr[last_below]:
                long_entry[i] = True
            else:
                short_entry[i] = True

    return (pd.Series(long_entry, index=ohlc.index),
            pd.Series(short_entry, index=ohlc.index))


# ============================================================
# Main
# ============================================================

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/BTCUSD_daily.csv'

    df = pd.read_csv(data_path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = [c[0] for c in df.columns]
    ohlc = df
    close_arr = ohlc['Close'].values

    print(f"Data: {len(ohlc)} bars, "
          f"{ohlc.index[0].date()} to {ohlc.index[-1].date()}")

    # S/R + RSI signal
    print("Computing S/R + RSI signals...", flush=True)
    sr_signal = compute_sr_signal(ohlc)
    sr_le = pd.Series((sr_signal == 1) & (np.roll(sr_signal, 1) != 1),
                      index=ohlc.index)
    sr_lx = pd.Series((sr_signal == -1) & (np.roll(sr_signal, 1) != -1),
                       index=ohlc.index)
    sr_se = sr_lx.copy()
    sr_sx = sr_le.copy()
    sr_le.iloc[0] = False
    sr_lx.iloc[0] = False
    sr_se.iloc[0] = False
    sr_sx.iloc[0] = False

    # Hawkes signals
    print("Computing Hawkes signals...", flush=True)
    h_long, h_short = compute_hawkes_signal(ohlc)

    # Union entries
    le = sr_le | h_long
    lx = sr_lx
    se = sr_se | h_short
    sx = sr_sx

    # Backtest
    print("Backtesting...\n")
    pf = vbt.Portfolio.from_signals(
        close=ohlc["Close"], entries=le, exits=lx,
        short_entries=se, short_exits=sx,
        init_cash=10_000, fees=0.001, freq="1D")

    bh = (close_arr[-1] / close_arr[0]) - 1
    nt = pf.trades.count()

    print("=" * 60)
    print("  S/R + RSI + Hawkes L+S Strategy")
    print("=" * 60)
    print(f"  Return:        {pf.total_return():.2%}")
    print(f"  Buy & Hold:    {bh:.2%}")
    print(f"  Alpha:         {pf.total_return() - bh:.2%}")
    print(f"  Sharpe:        {pf.sharpe_ratio():.4f}")
    print(f"  Sortino:       {pf.sortino_ratio():.4f}")
    print(f"  Max Drawdown:  {pf.max_drawdown():.2%}")
    print(f"  Trades:        {nt}")
    print(f"  Win Rate:      {pf.trades.win_rate():.1%}" if nt > 0 else "")
    print(f"  Profit Factor: {pf.trades.profit_factor():.2f}"
          if nt > 0 else "")
    print(f"  Final Value:   ${pf.value().iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
