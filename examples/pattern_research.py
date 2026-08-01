"""
Systematic research: integrate each TechnicalAnalysisAutomation pattern
with the trendline slope strategy and backtest on real BTC-USD daily data.

Tests (all long-only):
  0. Baseline: trendline slopes only (lb=15, sup=-0.001, res=0.003)
  1. + Directional Change filter
  2. + Head & Shoulders reversal signals
  3. + Flags/Pennants breakout confirmation
  4. + Market Profile S/R level filter
  5. + Harmonic Patterns entry signals
  6. Standalone: each pattern alone (no trendlines)
"""

import sys
import os
import numpy as np
import pandas as pd
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../TechnicalAnalysisAutomation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vectorbt as vbt

# Import pattern modules
from rolling_window import rw_top, rw_bottom
from perceptually_important import find_pips
from trendline_automation import fit_trendlines_single, optimize_slope, check_trend_line

# ============================================================
# Trendline engine
# ============================================================

def compute_slopes(high, low, close, lb):
    n = len(close)
    sup, res = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(lb - 1, n):
        s = slice(i - lb + 1, i + 1)
        h, l, c = high[s], low[s], close[s]
        x = np.arange(lb)
        cf = np.polyfit(x, c, 1)
        lp = cf[0] * x + cf[1]
        sup[i], _ = optimize_slope(True, (l - lp).argmin(), cf[0], l)
        res[i], _ = optimize_slope(False, (h - lp).argmax(), cf[0], h)
    return sup, res


# ============================================================
# Directional Change
# ============================================================

def directional_change(close, high, low, sigma):
    up_zig = True
    tmp_max, tmp_min = high[0], low[0]
    tmp_max_i, tmp_min_i = 0, 0
    tops, bottoms = [], []
    for i in range(len(close)):
        if up_zig:
            if high[i] > tmp_max:
                tmp_max, tmp_max_i = high[i], i
            elif close[i] < tmp_max - tmp_max * sigma:
                tops.append([i, tmp_max_i, tmp_max])
                up_zig = False
                tmp_min, tmp_min_i = low[i], i
        else:
            if low[i] < tmp_min:
                tmp_min, tmp_min_i = low[i], i
            elif close[i] > tmp_min + tmp_min * sigma:
                bottoms.append([i, tmp_min_i, tmp_min])
                up_zig = True
                tmp_max, tmp_max_i = high[i], i
    return tops, bottoms


def dc_signals(ohlc, sigma=0.03):
    """Generate buy signals at confirmed bottoms, sell at confirmed tops."""
    close = ohlc["Close"].values
    high = ohlc["High"].values
    low = ohlc["Low"].values
    tops, bottoms = directional_change(close, high, low, sigma)

    buy = np.zeros(len(ohlc), dtype=bool)
    sell = np.zeros(len(ohlc), dtype=bool)
    for b in bottoms:
        buy[b[0]] = True
    for t in tops:
        sell[t[0]] = True
    return pd.Series(buy, index=ohlc.index), pd.Series(sell, index=ohlc.index)


# ============================================================
# Head & Shoulders
# ============================================================

def hs_signals(ohlc, order=6):
    """Generate signals from H&S (bearish sell) and IH&S (bullish buy)."""
    from collections import deque

    data = np.log(ohlc["Close"].values)
    buy = np.zeros(len(ohlc), dtype=bool)
    sell = np.zeros(len(ohlc), dtype=bool)

    recent_extrema = deque(maxlen=5)
    recent_types = deque(maxlen=5)
    last_is_top = False
    hs_lock, ihs_lock = False, False

    def _check_hs(extrema_indices, data, i):
        l_shoulder, l_armpit, head, r_armpit = extrema_indices
        if i - r_armpit < 2: return False
        r_shoulder = r_armpit + data[r_armpit + 1: i].argmax() + 1
        if data[head] <= max(data[l_shoulder], data[r_shoulder]): return False
        r_mid = 0.5 * (data[r_shoulder] + data[r_armpit])
        l_mid = 0.5 * (data[l_shoulder] + data[l_armpit])
        if data[l_shoulder] < r_mid or data[r_shoulder] < l_mid: return False
        r_t = r_shoulder - head
        l_t = head - l_shoulder
        if r_t > 2.5 * l_t or l_t > 2.5 * r_t: return False
        neck_slope = (data[r_armpit] - data[l_armpit]) / (r_armpit - l_armpit)
        neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
        if data[i] > neck_val: return False
        return True

    def _check_ihs(extrema_indices, data, i):
        l_shoulder, l_armpit, head, r_armpit = extrema_indices
        if i - r_armpit < 2: return False
        r_shoulder = r_armpit + data[r_armpit + 1: i].argmin() + 1
        if data[head] >= min(data[l_shoulder], data[r_shoulder]): return False
        r_mid = 0.5 * (data[r_shoulder] + data[r_armpit])
        l_mid = 0.5 * (data[l_shoulder] + data[l_armpit])
        if data[l_shoulder] > r_mid or data[r_shoulder] > l_mid: return False
        r_t = r_shoulder - head
        l_t = head - l_shoulder
        if r_t > 2.5 * l_t or l_t > 2.5 * r_t: return False
        neck_slope = (data[r_armpit] - data[l_armpit]) / (r_armpit - l_armpit)
        neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
        if data[i] < neck_val: return False
        return True

    for i in range(len(data)):
        if rw_top(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(1)
            ihs_lock = False
            last_is_top = True
        if rw_bottom(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(-1)
            hs_lock = False
            last_is_top = False

        if len(recent_extrema) < 5:
            continue

        if last_is_top:
            ihs_ext = list(recent_extrema)[1:5]
            hs_ext = list(recent_extrema)[0:4]
        else:
            ihs_ext = list(recent_extrema)[0:4]
            hs_ext = list(recent_extrema)[1:5]

        if not hs_lock and _check_hs(hs_ext, data, i):
            sell[i] = True
            hs_lock = True
        if not ihs_lock and _check_ihs(ihs_ext, data, i):
            buy[i] = True
            ihs_lock = True

    return pd.Series(buy, index=ohlc.index), pd.Series(sell, index=ohlc.index)


# ============================================================
# Flags & Pennants
# ============================================================

def flag_signals(ohlc, order=10):
    """Generate signals from flag/pennant breakouts."""
    data = np.log(ohlc["Close"].values)
    buy = np.zeros(len(ohlc), dtype=bool)
    sell = np.zeros(len(ohlc), dtype=bool)

    last_bottom = -1
    last_top = -1

    for i in range(len(data)):
        if rw_top(data, i, order):
            last_top = i - order
            if last_bottom != -1:
                # Check bull flag
                tip_x = last_top
                base_x = last_bottom
                if tip_x > base_x and i - tip_x >= 5:
                    pole_h = data[tip_x] - data[base_x]
                    pole_w = tip_x - base_x
                    flag_w = i - tip_x
                    if flag_w <= pole_w * 0.5 and pole_h > 0:
                        flag_h = data[tip_x] - data[tip_x:i].min()
                        if flag_h <= pole_h * 0.75 and flag_w >= 3:
                            try:
                                sc, rc = fit_trendlines_single(data[tip_x:i])
                                resist_val = rc[1] + rc[0] * (flag_w + 1)
                                if data[i] > resist_val:
                                    buy[i] = True
                            except:
                                pass

        if rw_bottom(data, i, order):
            last_bottom = i - order
            if last_top != -1:
                # Check bear flag
                tip_x = last_bottom
                base_x = last_top
                if tip_x > base_x and i - tip_x >= 5:
                    pole_h = data[base_x] - data[tip_x]
                    pole_w = tip_x - base_x
                    flag_w = i - tip_x
                    if flag_w <= pole_w * 0.5 and pole_h > 0:
                        flag_h = data[tip_x:i].max() - data[tip_x]
                        if flag_h <= pole_h * 0.75 and flag_w >= 3:
                            try:
                                sc, rc = fit_trendlines_single(data[tip_x:i])
                                support_val = sc[1] + sc[0] * (flag_w + 1)
                                if data[i] < support_val:
                                    sell[i] = True
                            except:
                                pass

    return pd.Series(buy, index=ohlc.index), pd.Series(sell, index=ohlc.index)


# ============================================================
# Market Profile S/R
# ============================================================

def sr_signals(ohlc, lookback=120, atr_mult=3.0, prom_thresh=0.25):
    """Generate signals when price penetrates S/R levels from KDE market profile."""
    import scipy.stats
    import scipy.signal

    close = ohlc["Close"].values
    log_close = np.log(close)

    # Compute ATR on log prices manually (no pandas_ta dependency)
    log_high = np.log(ohlc["High"].values)
    log_low = np.log(ohlc["Low"].values)
    prev_close = np.roll(log_close, 1)
    prev_close[0] = log_close[0]
    tr = np.maximum(log_high - log_low,
                    np.maximum(np.abs(log_high - prev_close),
                               np.abs(log_low - prev_close)))
    atr = pd.Series(tr).rolling(lookback).mean().values

    buy = np.zeros(len(ohlc), dtype=bool)
    sell = np.zeros(len(ohlc), dtype=bool)

    for i in range(lookback, len(ohlc)):
        vals = log_close[i - lookback + 1: i + 1]
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        try:
            first_w = 0.01
            w_step = (1.0 - first_w) / len(vals)
            weights = first_w + np.arange(len(vals)) * w_step
            weights[weights < 0] = 0.0

            kernel = scipy.stats.gaussian_kde(vals, bw_method=a * atr_mult, weights=weights)
            min_v, max_v = vals.min(), vals.max()
            price_range = np.linspace(min_v, max_v, 200)
            pdf = kernel(price_range)
            peaks, _ = scipy.signal.find_peaks(pdf, prominence=np.max(pdf) * prom_thresh)
            levels = [np.exp(price_range[p]) for p in peaks]
        except:
            continue

        if i < 1:
            continue

        last_c = close[i - 1]
        curr_c = close[i]
        for level in levels:
            if curr_c > level and last_c <= level:
                buy[i] = True
            elif curr_c < level and last_c >= level:
                sell[i] = True

    return pd.Series(buy, index=ohlc.index), pd.Series(sell, index=ohlc.index)


# ============================================================
# Harmonic Patterns
# ============================================================

def harmonic_signals(ohlc, sigma=0.02, err_thresh=0.3):
    """Generate signals from XABCD harmonic patterns."""
    from math import log as mlog

    close = ohlc["Close"].values
    high = ohlc["High"].values
    low = ohlc["Low"].values

    # Get extremes via directional change
    tops, bottoms = directional_change(close, high, low, sigma)
    tops_df = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
    bottoms_df = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])
    tops_df['type'] = 1
    bottoms_df['type'] = -1
    extremes = pd.concat([tops_df, bottoms_df]).set_index('conf_i').sort_index()

    if len(extremes) < 4:
        return pd.Series(np.zeros(len(ohlc), dtype=bool), index=ohlc.index), \
               pd.Series(np.zeros(len(ohlc), dtype=bool), index=ohlc.index)

    extremes['seg_height'] = (extremes['ext_p'] - extremes['ext_p'].shift(1)).abs()
    extremes['retrace_ratio'] = extremes['seg_height'] / extremes['seg_height'].shift(1)

    PATTERNS = [
        ("Gartley", 0.618, [0.382, 0.886], [1.13, 1.618], 0.786),
        ("Bat", [0.382, 0.50], [0.382, 0.886], [1.618, 2.618], 0.886),
        ("Butterfly", 0.786, [0.382, 0.886], [1.618, 2.24], [1.27, 1.41]),
        ("Crab", [0.382, 0.618], [0.382, 0.886], [2.618, 3.618], 1.618),
        ("Cypher", [0.382, 0.618], [1.13, 1.41], [1.27, 2.00], 0.786),
    ]

    def get_err(actual, expected):
        if expected is None: return 0.0
        la = mlog(max(actual, 1e-10))
        if isinstance(expected, list):
            l0, l1 = mlog(expected[0]), mlog(expected[1])
            if l0 <= la <= l1: return 0.0
            return min(abs(la - l0), abs(la - l1)) * 2.0
        return abs(la - mlog(expected))

    buy = np.zeros(len(ohlc), dtype=bool)
    sell = np.zeros(len(ohlc), dtype=bool)

    first_conf = extremes.index[0]
    extreme_i = 0
    entry_taken = False

    for i in range(first_conf, len(ohlc)):
        if extreme_i + 1 < len(extremes) and extremes.index[extreme_i + 1] == i:
            entry_taken = False
            extreme_i += 1
        if entry_taken or extreme_i + 1 >= len(extremes) or extreme_i < 3:
            continue

        ext_type = extremes.iloc[extreme_i]['type']
        last_conf_i = extremes.index[extreme_i]

        if ext_type > 0:
            D_price = low[i]
            if last_conf_i + 1 >= i: continue
            if low[last_conf_i + 1:i].min() < D_price: continue
        else:
            D_price = high[i]
            if last_conf_i + 1 >= i: continue
            if high[last_conf_i + 1:i].max() > D_price: continue

        seg_h = extremes.iloc[extreme_i]['seg_height']
        if seg_h == 0: continue
        dc_retrace = abs(D_price - extremes.iloc[extreme_i]['ext_p']) / seg_h
        prev_seg_h = extremes.iloc[extreme_i - 2]['seg_height']
        if prev_seg_h == 0: continue
        xa_ad_retrace = abs(D_price - extremes.iloc[extreme_i - 2]['ext_p']) / prev_seg_h

        best_err = 1e30
        for name, xa_ab, ab_bc, bc_cd, xa_ad in PATTERNS:
            err = 0.0
            rr = extremes.iloc[extreme_i]['retrace_ratio']
            rr_prev = extremes.iloc[extreme_i - 1]['retrace_ratio']
            if np.isnan(rr) or np.isnan(rr_prev): continue
            err += get_err(rr, ab_bc)
            err += get_err(rr_prev, xa_ab)
            err += get_err(dc_retrace, bc_cd)
            err += get_err(xa_ad_retrace, xa_ad)
            best_err = min(best_err, err)

        if best_err <= err_thresh:
            entry_taken = True
            if ext_type > 0:
                buy[i] = True
            else:
                sell[i] = True

    return pd.Series(buy, index=ohlc.index), pd.Series(sell, index=ohlc.index)


# ============================================================
# Load data & run all strategies
# ============================================================

def load_ohlc(path='/tmp/BTCUSD_daily.csv'):
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = [c[0] for c in df.columns]
    return df


def backtest(ohlc, entries, exits, label, freq="1D"):
    kw = dict(close=ohlc["Close"], entries=entries, exits=exits, init_cash=10_000, fees=0.001, freq=freq)
    pf = vbt.Portfolio.from_signals(**kw)
    bh = (ohlc["Close"].iloc[-1] / ohlc["Close"].iloc[0]) - 1
    nt = pf.trades.count()
    sh = pf.sharpe_ratio()
    return dict(
        strategy=label, sharpe=sh if np.isfinite(sh) else 0.0,
        ret=pf.total_return(), bh=bh, alpha=pf.total_return() - bh,
        dd=pf.max_drawdown(), trades=nt,
        wr=pf.trades.win_rate() if nt > 0 else 0.0,
        pf=pf.trades.profit_factor() if nt > 0 else 0.0,
    )


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/BTCUSD_daily.csv'
    label = sys.argv[2] if len(sys.argv) > 2 else 'BTC-USD'
    ohlc = load_ohlc(data_path)
    print(f"{label} daily: {len(ohlc)} bars, {ohlc.index[0].date()} to {ohlc.index[-1].date()}")
    print()

    # Trendline baseline
    LOOKBACK = 15
    SUP_THRESH = -0.001
    RES_THRESH = 0.003
    log_h = np.log(ohlc["High"].values)
    log_l = np.log(ohlc["Low"].values)
    log_c = np.log(ohlc["Close"].values)

    print("Computing trendline slopes...", flush=True)
    t0 = time.time()
    sup_slopes, res_slopes = compute_slopes(log_h, log_l, log_c, LOOKBACK)
    support = pd.Series(sup_slopes, index=ohlc.index)
    resist = pd.Series(res_slopes, index=ohlc.index)
    trend_buy = (support > SUP_THRESH) & (support.shift(1) <= SUP_THRESH)
    trend_sell = (resist < RES_THRESH) & (resist.shift(1) >= RES_THRESH)
    print(f"  Done in {time.time()-t0:.1f}s")

    results = []

    # 0. Baseline: trendline only
    print("0. Baseline trendline...", flush=True)
    results.append(backtest(ohlc, trend_buy, trend_sell, "Trendline Only"))

    # 1. Directional Change standalone + combined
    print("1. Directional Change...", flush=True)
    t0 = time.time()
    dc_buy, dc_sell = dc_signals(ohlc, sigma=0.03)
    print(f"  Computed in {time.time()-t0:.1f}s (buy={dc_buy.sum()}, sell={dc_sell.sum()})")
    results.append(backtest(ohlc, dc_buy, dc_sell, "DC Standalone"))
    # Combined: trend entry confirmed by DC bottom nearby (within 5 bars)
    dc_buy_near = dc_buy.rolling(5, min_periods=1).max().astype(bool)
    combined_buy = trend_buy & dc_buy_near
    results.append(backtest(ohlc, combined_buy, trend_sell, "Trend + DC Filter"))

    # 2. Head & Shoulders
    print("2. Head & Shoulders...", flush=True)
    t0 = time.time()
    hs_buy, hs_sell = hs_signals(ohlc, order=6)
    print(f"  Computed in {time.time()-t0:.1f}s (buy={hs_buy.sum()}, sell={hs_sell.sum()})")
    results.append(backtest(ohlc, hs_buy, hs_sell, "H&S Standalone"))
    # Combined: use IH&S buy as entry, H&S sell OR trendline sell as exit
    combined_exit = trend_sell | hs_sell
    results.append(backtest(ohlc, trend_buy, combined_exit, "Trend + H&S Exit"))
    # Or: IH&S confirms trend entry
    hs_buy_near = hs_buy.rolling(10, min_periods=1).max().astype(bool)
    results.append(backtest(ohlc, trend_buy & hs_buy_near, trend_sell, "Trend + IH&S Confirm"))

    # 3. Flags & Pennants
    print("3. Flags & Pennants...", flush=True)
    t0 = time.time()
    flag_buy, flag_sell = flag_signals(ohlc, order=10)
    print(f"  Computed in {time.time()-t0:.1f}s (buy={flag_buy.sum()}, sell={flag_sell.sum()})")
    results.append(backtest(ohlc, flag_buy, flag_sell, "Flags Standalone"))
    # Combined: enter on bull flag OR trend signal
    results.append(backtest(ohlc, trend_buy | flag_buy, trend_sell | flag_sell, "Trend + Flags Union"))
    # Combined: enter only when both agree (within 5 bars)
    flag_buy_near = flag_buy.rolling(5, min_periods=1).max().astype(bool)
    results.append(backtest(ohlc, trend_buy & flag_buy_near, trend_sell, "Trend + Flags Confirm"))

    # 4. Market Profile S/R
    print("4. Market Profile S/R...", flush=True)
    t0 = time.time()
    sr_buy, sr_sell = sr_signals(ohlc, lookback=120, atr_mult=3.0, prom_thresh=0.25)
    print(f"  Computed in {time.time()-t0:.1f}s (buy={sr_buy.sum()}, sell={sr_sell.sum()})")
    results.append(backtest(ohlc, sr_buy, sr_sell, "S/R Standalone"))
    # Combined: trend entry near S/R breakout
    sr_buy_near = sr_buy.rolling(3, min_periods=1).max().astype(bool)
    results.append(backtest(ohlc, trend_buy & sr_buy_near, trend_sell, "Trend + S/R Filter"))
    # Combined: trend entry, exit at S/R sell
    results.append(backtest(ohlc, trend_buy, trend_sell | sr_sell, "Trend + S/R Exit"))

    # 5. Harmonic Patterns
    print("5. Harmonic Patterns...", flush=True)
    t0 = time.time()
    harm_buy, harm_sell = harmonic_signals(ohlc, sigma=0.02, err_thresh=0.3)
    print(f"  Computed in {time.time()-t0:.1f}s (buy={harm_buy.sum()}, sell={harm_sell.sum()})")
    results.append(backtest(ohlc, harm_buy, harm_sell, "Harmonic Standalone"))
    harm_buy_near = harm_buy.rolling(5, min_periods=1).max().astype(bool)
    results.append(backtest(ohlc, trend_buy & harm_buy_near, trend_sell, "Trend + Harmonic Confirm"))
    results.append(backtest(ohlc, trend_buy | harm_buy, trend_sell | harm_sell, "Trend + Harmonic Union"))

    # Print results
    print()
    print("=" * 130)
    print(f"{'Strategy':<28} {'Sharpe':>8} {'Return':>10} {'B&H':>10} {'Alpha':>10} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8} {'PF':>8}")
    print("-" * 130)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        print(f"{r['strategy']:<28} {r['sharpe']:>8.4f} {r['ret']:>9.2%} {r['bh']:>9.2%} {r['alpha']:>9.2%} {r['dd']:>8.2%} {r['trades']:>7} {r['wr']:>7.2%} {r['pf']:>8.2f}")
    print("=" * 130)


if __name__ == "__main__":
    main()
