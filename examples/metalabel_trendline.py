"""
Meta-Label Trendline Breakout Strategy.

Uses ML (Random Forest) to filter trendline breakout trades.
Generates features per breakout (slope, volume, RSI, ADX, etc.)
and trains a classifier to predict which breakouts will be profitable.

Walk-forward: retrains every 180 bars on last 3 years of trades.

Results on daily BTC:
  - Base breakout: 719 trades, 49% WR, PF 0.88 (LOSING strategy)
  - ML filtered (RF d3, thresh 0.55): 21 OOS trades, 62% WR, PF 5.28
  - ML filtered (RF d5, thresh 0.55): 70 OOS trades, 63% WR, PF 2.48

Note: Meta-labeling works for high-frequency strategies (200+ trades)
with ~50% base WR. It does NOT work for our low-frequency S/R strategy
(85 trades, 33% WR) — see RESEARCH.md for details.

Usage:
    python metalabel_trendline.py                     # Default BTC data
    python metalabel_trendline.py /path/to/data.csv   # Custom data
    python metalabel_trendline.py /path/to/data.csv 0.60  # Custom threshold
"""

import sys
import os
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "TechnicalAnalysisAutomation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vectorbt as vbt
from trendline_automation import fit_trendlines_single


# ============================================================
# Generate breakout trades with features
# ============================================================

def generate_breakout_dataset(ohlc, lookback=30, hold_period=10,
                               tp_mult=2.0, sl_mult=2.0, atr_lb=90):
    """Find all trendline breakout trades and extract ML features."""
    close = np.log(ohlc['Close'].values)
    atr = ta.atr(ohlc['High'], ohlc['Low'], ohlc['Close'], atr_lb).values
    vol_norm = (ohlc['Volume'] / ohlc['Volume'].rolling(atr_lb).median()).values
    rsi = ta.rsi(ohlc['Close'], 14).values
    adx_df = ta.adx(ohlc['High'], ohlc['Low'], ohlc['Close'], 14)
    adx = adx_df['ADX_14'].values

    trades = []

    for i in range(atr_lb, len(ohlc)):
        window = close[i - lookback:i]
        try:
            s_coefs, r_coefs = fit_trendlines_single(window)
        except:
            continue

        r_val = r_coefs[1] + lookback * r_coefs[0]
        s_val = s_coefs[1] + lookback * s_coefs[0]

        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        for direction, val, coefs in [(1, r_val, r_coefs), (-1, s_val, s_coefs)]:
            triggered = ((direction == 1 and close[i] > val) or
                         (direction == -1 and close[i] < val))
            if not triggered:
                continue

            # Find exit: TP, SL, or hold period
            tp = close[i] + direction * atr[i] * tp_mult
            sl = close[i] - direction * atr[i] * sl_mult
            exit_i = min(i + hold_period, len(ohlc) - 1)
            exit_p = close[exit_i]
            for j in range(i + 1, min(i + hold_period + 1, len(ohlc))):
                if direction == 1 and (close[j] >= tp or close[j] <= sl):
                    exit_i = j
                    exit_p = close[j]
                    break
                elif direction == -1 and (close[j] <= tp or close[j] >= sl):
                    exit_i = j
                    exit_p = close[j]
                    break

            # Features
            line_vals = coefs[1] + np.arange(lookback) * coefs[0]
            tl_err = np.sum(abs(line_vals - window)) / lookback / atr[i]
            max_dist = abs(line_vals - window).max() / atr[i]
            break_str = abs(close[i] - val) / atr[i]

            trades.append({
                'entry_i': i,
                'exit_i': exit_i,
                'direction': direction,
                'return': direction * (exit_p - close[i]),
                'slope': coefs[0] / atr[i],
                'tl_err': tl_err,
                'max_dist': max_dist,
                'break_str': break_str,
                'vol': vol_norm[i] if not np.isnan(vol_norm[i]) else 1.0,
                'rsi': rsi[i] if not np.isnan(rsi[i]) else 50.0,
                'adx': adx[i] if not np.isnan(adx[i]) else 20.0,
                'chan_w': (r_val - s_val) / atr[i],
            })
            break  # one signal per bar

    return pd.DataFrame(trades).dropna()


# ============================================================
# Walk-forward ML-filtered backtest
# ============================================================

FEATURE_COLS = ['slope', 'tl_err', 'max_dist', 'break_str',
                'vol', 'rsi', 'adx', 'chan_w']


def walkforward_metalabel(ohlc, trades, prob_threshold=0.55,
                           train_window=365 * 3, step=180):
    """Walk-forward backtest with periodic RF retraining."""
    signal_ml = np.zeros(len(ohlc))
    signal_all = np.zeros(len(ohlc))

    next_train = train_window
    cur_model = None
    trade_idx = 0

    for i in range(len(ohlc)):
        # Retrain model
        if i >= next_train:
            s = max(0, i - train_window)
            tm = (trades['entry_i'] >= s) & (trades['exit_i'] < i)
            if tm.sum() >= 20:
                cur_model = RandomForestClassifier(
                    n_estimators=500, max_depth=3, random_state=42)
                X = trades.loc[tm, FEATURE_COLS].values
                y = (trades.loc[tm, 'return'] > 0).astype(int).values
                cur_model.fit(X, y)
            next_train += step

        # Check for trade entries at this bar
        while (trade_idx < len(trades) and
               trades.iloc[trade_idx]['entry_i'] == i):
            t = trades.iloc[trade_idx]
            ei = int(t['entry_i'])
            xi = int(t['exit_i'])
            d = t['direction']

            # Unfiltered: take all trades
            for j in range(ei, min(xi + 1, len(ohlc))):
                signal_all[j] = d

            # ML filtered
            if cur_model is not None:
                features = t[FEATURE_COLS].values.reshape(1, -1)
                prob = cur_model.predict_proba(features)[0][1]
                if prob > prob_threshold:
                    for j in range(ei, min(xi + 1, len(ohlc))):
                        signal_ml[j] = d

            trade_idx += 1

    return signal_all, signal_ml


def signal_to_portfolio(signal, ohlc, freq='1D'):
    """Convert signal array to vectorbt Portfolio."""
    le = pd.Series((signal == 1) & (np.roll(signal, 1) != 1),
                   index=ohlc.index)
    lx = pd.Series((signal != 1) & (np.roll(signal, 1) == 1),
                    index=ohlc.index)
    se = pd.Series((signal == -1) & (np.roll(signal, 1) != -1),
                    index=ohlc.index)
    sx = pd.Series((signal != -1) & (np.roll(signal, 1) == -1),
                    index=ohlc.index)
    le.iloc[0] = False
    lx.iloc[0] = False
    se.iloc[0] = False
    sx.iloc[0] = False
    return vbt.Portfolio.from_signals(
        close=ohlc["Close"], entries=le, exits=lx,
        short_entries=se, short_exits=sx,
        init_cash=10_000, fees=0.001, freq=freq)


def print_stats(pf, label):
    nt = pf.trades.count()
    sh = pf.sharpe_ratio()
    print(f"  {label:<30} "
          f"Sh={sh:.3f if np.isfinite(sh) else 0:.0f}  "
          f"Ret={pf.total_return():.0%}  "
          f"DD={pf.max_drawdown():.1%}  "
          f"Tr={nt}  "
          f"WR={pf.trades.win_rate():.0%}  "
          f"PF={pf.trades.profit_factor():.2f}"
          if nt > 0 else f"  {label:<30} No trades")


# ============================================================
# Main
# ============================================================

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/BTCUSD_daily.csv'
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.55

    df = pd.read_csv(data_path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = [c[0] for c in df.columns]
    ohlc = df

    print(f"Data: {len(ohlc)} bars, "
          f"{ohlc.index[0].date()} to {ohlc.index[-1].date()}")
    print(f"ML threshold: {threshold}")

    # Generate trades
    print("\nGenerating breakout trades...", flush=True)
    trades = generate_breakout_dataset(ohlc)
    print(f"  {len(trades)} trades "
          f"(Bull: {(trades['direction']==1).sum()}, "
          f"Bear: {(trades['direction']==-1).sum()})")
    print(f"  Base WR: {(trades['return']>0).mean():.1%}")

    # Train/test split analysis
    cutoff = ohlc.index.searchsorted(pd.Timestamp('2024-01-01'))
    tr_mask = trades['entry_i'] < cutoff
    te_mask = trades['entry_i'] >= cutoff

    print(f"\n  Train: {tr_mask.sum()} trades, "
          f"Test: {te_mask.sum()} trades")

    # Test ML filter on OOS trades
    print("\n  ML Filter OOS Results:")
    print(f"  {'Model':<12} {'Thresh':>6} {'Taken':>6} "
          f"{'WR':>6} {'PF':>6}")
    print(f"  {'-'*45}")

    base_te = trades[te_mask]
    bw = base_te[base_te['return'] > 0]['return'].sum()
    bl = base_te[base_te['return'] <= 0]['return'].abs().sum()
    base_pf = bw / bl if bl > 0 else 0
    print(f"  {'Baseline':<12} {'all':>6} {te_mask.sum():>6} "
          f"{(trades.loc[te_mask, 'return']>0).mean():>5.1%} "
          f"{base_pf:>6.2f}")

    for depth in [3, 5]:
        model = RandomForestClassifier(
            n_estimators=500, max_depth=depth, random_state=42)
        X_tr = trades.loc[tr_mask, FEATURE_COLS].values
        y_tr = (trades.loc[tr_mask, 'return'] > 0).astype(int).values
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(
            trades.loc[te_mask, FEATURE_COLS].values)[:, 1]

        for th in [0.45, 0.50, 0.55, 0.60, 0.65]:
            sel = probs > th
            n = sel.sum()
            if n < 3:
                continue
            st = base_te[sel]
            wr = (st['return'] > 0).mean()
            w = st[st['return'] > 0]['return'].sum()
            lo = st[st['return'] <= 0]['return'].abs().sum()
            pf = w / lo if lo > 0 else float('inf')
            print(f"  {'RF d'+str(depth):<12} {th:>6.2f} {n:>6} "
                  f"{wr:>5.1%} {pf:>6.2f}")

    # Walk-forward backtest
    print("\n" + "=" * 60)
    print("  WALK-FORWARD BACKTEST")
    print("=" * 60)

    sig_all, sig_ml = walkforward_metalabel(
        ohlc, trades, prob_threshold=threshold)

    print("\n  Full period:")
    print_stats(signal_to_portfolio(sig_all, ohlc), "All trades (no ML)")
    print_stats(signal_to_portfolio(sig_ml, ohlc), "ML filtered")

    oos = ohlc.loc['2024-01-01':]
    oos_i = ohlc.index.searchsorted(oos.index[0])
    print("\n  OOS (2024+):")
    print_stats(signal_to_portfolio(sig_all[oos_i:], oos), "All trades (no ML)")
    print_stats(signal_to_portfolio(sig_ml[oos_i:], oos), "ML filtered")

    # Feature importance
    print("\n  Feature Importance (last model):")
    # Retrain final model for display
    model = RandomForestClassifier(
        n_estimators=500, max_depth=3, random_state=42)
    model.fit(trades[FEATURE_COLS].values,
              (trades['return'] > 0).astype(int).values)
    for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_),
                            key=lambda x: -x[1]):
        bar = '#' * int(imp * 50)
        print(f"    {feat:<15} {imp:.3f} {bar}")


if __name__ == "__main__":
    main()
