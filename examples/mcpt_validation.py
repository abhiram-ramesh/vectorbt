"""
Monte Carlo Permutation Test (MCPT) for strategy validation.

Tests whether a strategy's performance is statistically significant
or just luck by shuffling OHLC bars and comparing real vs permuted performance.

Based on: https://github.com/neurotrader888/mcpt

Usage:
    python mcpt_validation.py [n_permutations]

    Default: 100 permutations (~35 seconds)
    For publication-grade: 1000 permutations (~6 minutes)
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import pandas_ta as ta
import scipy.stats as st_stats
import scipy.signal as sig_mod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vectorbt as vbt


# ============================================================
# Bar Permutation (preserves intrabar OHLC structure)
# ============================================================

def get_permutation(ohlc, start_index=0, seed=None):
    """Shuffle OHLC bars while preserving intrabar structure.

    Each bar's open-high-low-close relationships are preserved,
    but the order of bars is randomized. This destroys temporal
    patterns (trends, S/R levels) while keeping the return
    distribution identical to real data.
    """
    np.random.seed(seed)
    n = len(ohlc)
    log_bars = np.log(ohlc[['Open', 'High', 'Low', 'Close']].values)

    pi = start_index + 1
    pn = n - pi

    start_bar = log_bars[start_index].copy()

    # Prices relative to bar's open
    r_o = np.roll(log_bars[:, 0] - np.roll(log_bars[:, 3], 1), 0)
    r_o[0] = 0
    r_h = log_bars[:, 1] - log_bars[:, 0]
    r_l = log_bars[:, 2] - log_bars[:, 0]
    r_c = log_bars[:, 3] - log_bars[:, 0]

    # Shuffle intrabar and gap separately
    idx = np.arange(pn)
    perm1 = np.random.permutation(idx)
    perm2 = np.random.permutation(idx)

    rh = r_h[pi:][perm1]
    rl = r_l[pi:][perm1]
    rc = r_c[pi:][perm1]
    ro = r_o[pi:][perm2]

    # Reconstruct price series
    perm_bars = np.zeros((n, 4))
    perm_bars[:start_index] = log_bars[:start_index]
    perm_bars[start_index] = start_bar

    for i in range(pi, n):
        k = i - pi
        perm_bars[i, 0] = perm_bars[i - 1, 3] + ro[k]
        perm_bars[i, 1] = perm_bars[i, 0] + rh[k]
        perm_bars[i, 2] = perm_bars[i, 0] + rl[k]
        perm_bars[i, 3] = perm_bars[i, 0] + rc[k]

    perm_bars = np.exp(perm_bars)
    return pd.DataFrame(perm_bars, index=ohlc.index,
                        columns=['Open', 'High', 'Low', 'Close'])


# ============================================================
# S/R + RSI Strategy (our production strategy)
# ============================================================

def run_strategy(ohlc):
    """Run S/R + RSI filter L+S strategy on OHLC data.

    Returns profit factor and Sharpe ratio.
    """
    close_arr = ohlc['Close'].values
    log_c = np.log(close_arr)

    # S/R levels via KDE market profile
    sr_df = pd.DataFrame({
        'high': ohlc['High'], 'low': ohlc['Low'], 'close': ohlc['Close']
    }, index=ohlc.index)
    atr_sr = ta.atr(np.log(sr_df['high']), np.log(sr_df['low']),
                    np.log(sr_df['close']), 180)

    all_levels = [None] * len(ohlc)
    for i in range(180, len(ohlc)):
        a = atr_sr.iloc[i]
        if pd.isna(a) or a <= 0:
            continue
        try:
            weights = np.ones(180)
            kernel = st_stats.gaussian_kde(log_c[i - 179:i + 1],
                                           bw_method=a * 3.0, weights=weights)
            min_v, max_v = log_c[i - 179:i + 1].min(), log_c[i - 179:i + 1].max()
            step = (max_v - min_v) / 200
            if step <= 0:
                continue
            pr = np.arange(min_v, max_v, step)
            pdf = kernel(pr)
            peaks, _ = sig_mod.find_peaks(pdf, prominence=np.max(pdf) * 0.25)
            all_levels[i] = [np.exp(pr[p]) for p in peaks]
        except:
            continue

    # Generate S/R signal
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
    rsi = ta.rsi(ohlc['Close'], 14).values
    f = sr_signal.copy()
    for i in range(1, len(f)):
        if f[i] != f[i - 1]:
            if f[i] == 1.0 and (np.isnan(rsi[i]) or rsi[i] < 50):
                f[i] = f[i - 1]
            elif f[i] == -1.0 and (np.isnan(rsi[i]) or rsi[i] > 50):
                f[i] = f[i - 1]

    # Compute metrics
    r = np.log(ohlc['Close']).diff().shift(-1).values
    sig_r = f * r
    wins = np.nansum(sig_r[sig_r > 0])
    losses = np.nansum(np.abs(sig_r[sig_r < 0]))
    pf = wins / losses if losses > 0 else 0

    daily_r = sig_r[~np.isnan(sig_r)]
    sharpe = (daily_r.mean() / daily_r.std() * np.sqrt(365)
              if daily_r.std() > 0 else 0)

    return pf, sharpe


# ============================================================
# Main
# ============================================================

def main():
    n_perms = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    data_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/BTCUSD_daily.csv'

    # Load data
    df = pd.read_csv(data_path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = [c[0] for c in df.columns]
    ohlc = df

    print("=" * 70)
    print("  MONTE CARLO PERMUTATION TEST (MCPT)")
    print("  Strategy: S/R + RSI Filter L+S")
    print(f"  Data: {len(ohlc)} bars, "
          f"{ohlc.index[0].date()} to {ohlc.index[-1].date()}")
    print(f"  Permutations: {n_perms}")
    print("=" * 70)

    # Real performance
    print("\nRunning on real data...", flush=True)
    real_pf, real_sharpe = run_strategy(ohlc)
    print(f"  Real Profit Factor: {real_pf:.4f}")
    print(f"  Real Sharpe:        {real_sharpe:.4f}")

    # Permutations
    print(f"\nRunning {n_perms} permutations...", flush=True)
    perm_pfs = []
    perm_sharpes = []
    pf_better = 1
    sharpe_better = 1
    t0 = time.time()

    for i in range(n_perms):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_perms - i - 1)
            print(f"  {i+1}/{n_perms} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)...",
                  flush=True)

        perm_ohlc = get_permutation(ohlc, start_index=180, seed=i)
        perm_ohlc['Volume'] = ohlc['Volume'].values

        perm_pf, perm_sharpe = run_strategy(perm_ohlc)
        perm_pfs.append(perm_pf)
        perm_sharpes.append(perm_sharpe)

        if perm_pf >= real_pf:
            pf_better += 1
        if perm_sharpe >= real_sharpe:
            sharpe_better += 1

    total_time = time.time() - t0
    pf_pval = pf_better / (n_perms + 1)
    sharpe_pval = sharpe_better / (n_perms + 1)

    # Results
    print(f"\nCompleted in {total_time:.0f}s")
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"  Profit Factor:")
    print(f"    Real:       {real_pf:.4f}")
    print(f"    Perm mean:  {np.mean(perm_pfs):.4f}")
    print(f"    Perm max:   {np.max(perm_pfs):.4f}")
    print(f"    P-Value:    {pf_pval:.4f}  "
          f"{'*** SIGNIFICANT' if pf_pval < 0.05 else 'NOT significant'}")
    print()
    print(f"  Sharpe Ratio:")
    print(f"    Real:       {real_sharpe:.4f}")
    print(f"    Perm mean:  {np.mean(perm_sharpes):.4f}")
    print(f"    Perm max:   {np.max(perm_sharpes):.4f}")
    print(f"    P-Value:    {sharpe_pval:.4f}  "
          f"{'*** SIGNIFICANT' if sharpe_pval < 0.05 else 'NOT significant'}")
    print()
    print(f"  Permutation PF distribution:")
    for q in [5, 25, 50, 75, 95]:
        print(f"    {q}th percentile: {np.percentile(perm_pfs, q):.4f}")
    print(f"    Real:            {real_pf:.4f}")


if __name__ == "__main__":
    main()
