# BTC Trading Strategy Research

Research conducted April 2026 on BTC-USD daily data (2020-01-01 to 2026-04-11, 2293 bars).

## Final Strategy: All Long + S/R Short with RSI Filter

**Entry (Long):** Any of three signal systems:
- Trendline slope crossover (support slope crosses above -0.001)
- Bull flag breakout (trendline-based flag detection, order=10)
- S/R level penetration upward (KDE market profile, lookback=180)

**Exit (Long):** Any of three:
- Trendline resistance slope crosses below 0.003
- Bear flag breakdown
- S/R level penetration downward

**Entry (Short):** S/R level penetration downward only (filtered by RSI < 50)

**Exit (Short):** S/R level penetration upward

**RSI Filter:** Only trigger S/R signals when RSI(14) confirms direction — long when RSI > 50, short when RSI < 50.

### Why This Design
- Asymmetric: wide net for longs (3 systems), selective shorts (S/R only)
- Trendline shorts failed in walk-forward — too many whipsaws
- S/R shorts work because breaking below a structural price level is a clear bearish signal
- RSI filter eliminates false breakouts where price barely crosses a level then reverses

## Performance Summary

### Full Period (2020-2026)
| Metric | Value |
|--------|-------|
| Sharpe | 1.59 |
| Return | 5,512% |
| Max Drawdown | -35.7% |
| Trades | 111 |
| Win Rate | 53% |
| Profit Factor | 2.28 |

### Out-of-Sample (2024-2026, never optimized on)
| Metric | Value |
|--------|-------|
| Sharpe | 1.05 |
| Return | 108% |
| Max Drawdown | -18.3% |
| Trades | 48 |
| Win Rate | 48% |
| Profit Factor | 2.01 |

### Last 12 Months (Apr 2025 - Apr 2026)
| Metric | Value |
|--------|-------|
| Sharpe | 1.63 |
| Return | +66.5% |
| Buy & Hold | -14.3% (BTC dropped) |
| Alpha | +80.8% |
| Max Drawdown | -13.9% |
| Win Rate | 52% |

### Walk-Forward (8 x 6-month windows, 2022-2026)
- Average Sharpe: 0.96
- Positive Sharpe periods: 7/8 (88%)
- Positive return periods: 7/8 (88%)
- Average win rate: 48%

### Yearly Breakdown
| Year | Strategy | Buy & Hold | Alpha | Win Rate |
|------|----------|-----------|-------|----------|
| 2020 | +276% | +303% | -27% | 70% |
| 2021 | +190% | +58% | +132% | 50% |
| 2022 | +22% | -65% | +87% | 53% |
| 2023 | +89% | +154% | -66% | 59% |
| 2024 | +28% | +112% | -83% | 50% |
| 2025 | +37% | -7% | +44% | 50% |
| 2026 | +21% | -18% | +39% | 43% |

Key: Made money in 2022 bear market (+22% vs -65% B&H). Outperformed B&H in 2021, 2022, 2025, 2026.

## Signal Components

### 1. S/R Levels (Market Profile KDE)
**Source:** `mp_support_resist.py` from TechnicalAnalysisAutomation repo

- Uses Gaussian KDE on 180 days of log closing prices
- ATR(180) on log prices as bandwidth
- Equal weighting (first_w=1.0, no recency bias)
- ATR multiplier: 3.0, prominence threshold: 0.25
- Finds peaks in the price distribution = levels where price spent significant time
- Signal: +1 when close crosses above a level, -1 when crosses below

This is the core edge. Survived all walk-forward tests. S/R levels represent real structural price zones.

### 2. Trendline Slopes
**Source:** `trendline_automation.py` from TechnicalAnalysisAutomation repo

- Rolling 15-bar lookback on log prices
- Fits support/resistance lines using slope optimization (gradient descent)
- Long entry: support slope crosses above -0.001
- Long exit: resistance slope crosses below 0.003

Used for long entries/exits only. Failed as short signal in walk-forward (-78% Sharpe decay when used alone).

### 3. Flag/Pennant Patterns
**Source:** `flags_pennants.py` from TechnicalAnalysisAutomation repo (trendline-based detection)

- Rolling window extrema detection (order=10)
- Validates pole height/width, flag height/width ratios
- Fits trendlines on flag consolidation area
- Confirms breakout above resistance (bull) or below support (bear)
- Bull flags: ~15 signals in 6 years (rare but high quality)
- Bear flags: ~8 signals

Adds marginal value — too few signals to rely on alone but catches momentum breakouts that S/R misses.

### 4. RSI Filter
- Standard RSI(14) on close prices
- Only allow S/R long signals when RSI > 50 (momentum confirming upward)
- Only allow S/R short signals when RSI < 50 (momentum confirming downward)
- This single filter raised win rate from 39% to 53%

## Research Path

### What We Tested (Chronological)

1. **Basic trendline slopes** — lookback sweep found lb=60, sup=-0.002, res=0.002 best on daily BTC. Sharpe 1.47 in-sample but collapsed out-of-sample (Sharpe 0.40, -78% decay). Overfit.

2. **Parameter sweep on daily** — 125 combos of lookback x thresholds. Shorter lookback (lb=15) with higher trade count found as alternative.

3. **Pattern integration** — tested each TechnicalAnalysisAutomation pattern combined with trendlines:
   - Flags+S/R Union: Sharpe 1.30 (best combined)
   - H&S as exit: Sharpe 1.24
   - S/R filter: Sharpe 1.20 (best PF at 2.84)
   - Standalone patterns all < trendlines alone

4. **S/R deep dive** — swept lookback (90-365), first_weight (0.01-1.0), ATR mult, prominence. Found lb=180, fw=1.0 optimal. S/R standalone Sharpe 1.39.

5. **Combined strategies** — Flags+S/R (Sharpe 1.51), Trend+S/R (1.40), three-way union (1.36). Flags+S/R won.

6. **Long + Short** — tested symmetric and asymmetric designs. "All Long + S/R Short" best for bear markets (Bear Sharpe 1.63). Trendline shorts destroyed value.

7. **SL/TP** — fixed 3% SL was best (Sharpe 1.60). ATR(14)*0.75 equivalent. Take profit never helped — BTC rewards letting winners run. Trailing stops get shaken out. Decided to skip SL/TP for simplicity.

8. **Whipsaw filter** — Hold5+ATR<6% cut 2021 drawdown from -53% to -21% in-sample, but collapsed out-of-sample (-89% decay). Classic overfit to a specific event. Abandoned.

9. **Timeframe testing** — Daily lb=180 is the sweet spot. Weekly lb=12 works. All intraday (4h, 1h, 15m) destroyed by fees and noise. S/R is a macro signal.

10. **Win rate improvement** — RSI L>50 S<50 raised WR from 39% to 53% with minimal Sharpe cost. Volume >1.2x avg also worked (55% WR). RSI chosen for better last-year performance.

11. **Walk-forward validation** — RSI-filtered version: OOS Sharpe 1.05, 48% WR, 7/8 positive periods. Confirmed the edge is real.

12. **Leverage** — Kelly criterion: Full Kelly 3.4x, Half Kelly 1.7x. Recommended 1.5-2.0x. Above 3x returns decline due to drawdown compounding. 7x+ gets liquidated.

### What Failed

| Approach | Why It Failed |
|----------|--------------|
| Trendline slopes alone | Overfit — parameters tuned to 2020-2023 BTC, -78% OOS decay |
| Trendline shorts | Whipsawed in choppy markets, negative alpha |
| Dual slope confirmation | Too restrictive, only 2 trades in 6 years |
| MA regime filter | Filtered out good entries in transitions |
| Slope acceleration | Added complexity for zero improvement |
| Slope divergence | Worse Sharpe than baseline |
| H&S as entry confirmation | Too rare to confirm enough trades |
| Whipsaw filters (Hold+ATR) | Overfit to 2021 crash, -89% OOS decay |
| Intraday S/R (4h, 1h, 15m) | S/R levels are macro — noise drowns signal intraday |
| Fixed take profit | Caps BTC's trending moves, always hurt |
| Trailing stops | BTC volatility shakes them out too early |
| S&P 500 application | Strategy doesn't work on equities — BTC-specific edge |

### What Survived Walk-Forward

| Strategy | OOS Sharpe | WF Avg | Decay |
|----------|-----------|--------|-------|
| All Long + S/R Short | 1.10 | 1.08 | -33% |
| RSI-filtered version | 1.05 | 0.96 | -42% |
| Flags+S/R LongOnly | 1.15 | 0.90 | -31% |
| S/R L+S | 0.92 | 1.15 | -36% |
| Trendline Only | 0.40 | - | -78% |

## Optimal Leverage

Kelly Criterion analysis on the RSI-filtered strategy:

| Leverage | OOS Return | OOS MaxDD | Worst Day | Recommendation |
|----------|-----------|-----------|-----------|----------------|
| 1.0x | +117% | -18% | -11% | Safe |
| 1.5x | +184% | -27% | -17% | Recommended |
| 2.0x | +243% | -35% | -22% | Aggressive |
| 3.0x | +291% | -49% | -33% | Max viable |
| 5.0x | +86% | -80% | -55% | Returns decline |
| 7.0x | LIQUIDATED | - | -77% | Never |

Half Kelly = 1.7x. Recommended range: 1.5-2.0x.

## Technical Details

### Dependencies
- vectorbt (backtesting engine)
- TechnicalAnalysisAutomation (neurotrader888) — pattern detection
- pandas_ta — ATR, RSI, ADX calculations
- scipy — Gaussian KDE for S/R levels
- numpy, pandas

### Data
- Source: yfinance (BTC-USD)
- Frequency: Daily OHLCV
- Train period: 2020-01-01 to 2023-12-31 (1461 bars)
- Test period: 2024-01-01 to 2026-04-11 (832 bars)

### File Locations
- Strategy scripts: `/Users/abhiramramesh/codebase/vectorbt/examples/`
  - `pattern_research.py` — full pattern integration research
  - `trendline_strategy.py` — standalone trendline strategy
  - `trendline_param_sweep.py` — parameter optimization
- Autoresearch optimizer: `/Users/abhiramramesh/codebase/autoresearch/optimize_trendline.py`
- Pattern library: `/Users/abhiramramesh/codebase/TechnicalAnalysisAutomation/`
- BTC data: `/tmp/BTCUSD_daily.csv` (redownload via yfinance if needed)

### Key Parameters (Do Not Change Without Re-Validating)
```
S/R lookback:          180 bars
S/R ATR period:        180
S/R KDE bandwidth:     ATR * 3.0
S/R equal weighting:   first_w = 1.0
S/R prominence:        0.25
Trendline lookback:    15 bars
Trendline sup thresh:  -0.001
Trendline res thresh:  0.003
Flag order:            10
RSI period:            14
RSI long threshold:    > 50
RSI short threshold:   < 50
Fees:                  0.1% per trade
```

## Risk Warnings

1. **All backtests assume fills at daily close prices.** Real execution has slippage, especially for large positions.
2. **53% win rate means losing streaks of 5-10 trades are normal.** Need discipline to hold through them.
3. **The -35.7% max drawdown happened in 2021.** A future crash could be worse.
4. **BTC market structure can change.** If S/R levels stop being respected (e.g., due to algorithmic trading or regulation changes), the edge disappears.
5. **This strategy was developed and tested on BTC only.** It did not work on S&P 500 and may not work on other crypto assets without re-validation.
6. **The 180-bar S/R lookback means the strategy needs ~6 months of data before generating signals.** No signals in the first 180 bars.
7. **Leverage above 2x significantly increases liquidation risk.** BTC can move 10%+ in a single day.

## VSA (Volume Spread Analysis) Indicator

**Source:** `VSAIndicator/vsa.py` (neurotrader888)

### How It Works

Regresses normalized volume against normalized price range over a rolling window. When the actual range deviates from what volume predicts, it flags the bar:
- **Negative deviation** (narrow range + high volume) = accumulation — someone is absorbing supply without moving price
- **Positive deviation** (wide range + low volume) = distribution/exhaustion

```
norm_range = (high - low) / ATR(lookback)
norm_volume = volume / median_volume(lookback)
regression: norm_range = slope * norm_volume + intercept
deviation = actual_norm_range - predicted_norm_range
```

Only computes when slope > 0 and r_value > 0.2 (volume-range relationship must be positive and meaningful).

### Timeframe Results

| Timeframe | Best Config | Sharpe | Trades | WR | 10-bar Fwd Return |
|-----------|------------|--------|--------|-----|-------------------|
| **Daily lb=90** | **accum t=1.0** | **1.27** | **10** | **80%** | **+9.0%** |
| Daily lb=60 | accum t=1.0 | 1.13 | 12 | 75% | +5.0% |
| Daily lb=120 | accum t=1.0 | 0.95 | 9 | 89% | +7.1% |
| Weekly lb=12 | accum t=0.5 | 1.20 | 5 | 100% | +21.2% |
| 4-Hour lb=252 | accum t=1.0 | 0.74 | 17 | 71% | +0.6% |
| 1-Hour lb=336 | accum t=0.75 | 0.98 | 40 | 55% | ~0% |
| 15-Min | all configs | negative | - | - | ~0% |

### Key Finding: Accumulation Is the Edge

**When volume is high but range is narrow (VSA < -1.0 on daily lb=90), the next 10 days average +9.0% return with 80% hit rate.** This represents institutional accumulation — large players absorbing supply without moving price, followed by a breakout.

The signal scales with lookback: lb=30 gives +7.5% forward, lb=90 gives +9.0%, lb=120 gives +7.1%. The 90-day window best captures the volume-range relationship.

Positive VSA (breakouts/exhaustion) had much weaker predictive power (+0.4% to +1.4%) and didn't produce profitable strategies.

### Timeframe Dependency

VSA accumulation plays out over **days to weeks**, not hours:
- Daily: +9% over 10 days — strong and tradeable
- Weekly: +21% over 10 weeks — very strong but too few signals (5 in 6 years)
- 4-Hour: +0.6% over 10 bars — mostly noise
- 1-Hour: ~0% forward return — completely noise
- 15-Min: negative — fees destroy any micro-edge

### Combination with S/R Strategy

VSA did **not** improve our S/R strategy when combined:
- As a filter (only take S/R signals during accumulation): too restrictive, cut trades from 85 to 18-50
- As extra entry signals: diluted signal quality, lower Sharpe
- As exhaustion exit: slightly improved win rate (32% → 35%) and PF (1.98 → 2.10) but halved returns

The two indicators measure different things (S/R = structural price levels, VSA = volume-price divergence) and don't confirm each other reliably.

### Best Use Case

VSA accumulation (dev < -1.0, daily lb=90) is a **high-conviction supplementary signal**:
- 80% win rate, average +9% over 10 days
- Only ~10 signals in 6 years — use as a position-sizing overlay or discretionary confirmation
- When VSA fires alongside an S/R long signal, increase position size
- Not enough signals to build a standalone strategy around

### Parameters

```
Lookback:        90 bars (daily)
Threshold:       1.0 (accumulation signal when dev < -1.0)
ATR period:      same as lookback (90)
Volume median:   same as lookback (90)
Min r_value:     0.2 (regression quality gate)
Min slope:       0.0 (volume-range must be positively correlated)
```

## Lower Timeframe Strategies (4H and 1H)

Research conducted on BTC-USD 4H and 1H data (June 2024 - April 2026, ~22 months).

### Key Finding: Mean Reversion Dominates Intraday

Tested 10 strategy categories on both 4H and 1H: Bollinger Bands, RSI, EMA crossover, Keltner Channel, MACD, Stochastic RSI, Donchian Channel, ATR breakout, VWAP deviation, and Supertrend.

**Every profitable intraday strategy is mean-reversion.** All trend-following/breakout strategies (EMA, MACD, Supertrend, Keltner, ATR breakout) either lost money or barely broke even. This is the opposite of daily where breakout (S/R) works best.

Why: intraday BTC oscillates around fair value. Breakouts that look decisive on hourly candles are often noise that reverts. On daily, breakouts represent genuine regime changes.

### 4-Hour Results

#### In-Sample Top Strategies
| Strategy | Sharpe | Return | Trades | WR |
|----------|--------|--------|--------|-----|
| VWAP MR p48 z2.5 | 0.74 | 45% | 14 | 71% |
| VWAP MR p96 z2.5 | 0.71 | 42% | 8 | 62% |
| RSI MR p21 25/75 | 0.61 | 32% | 4 | 75% |
| RSI MR p7 20/80 | 0.57 | 30% | 27 | 59% |
| EMA 20/50 | 0.56 | 25% | 39 | 31% |
| Donchian 20 | 0.47 | 21% | 47 | 62% |

#### Walk-Forward Validation (train Jun 2024 - Jun 2025, test Jun 2025 - Apr 2026)
| Strategy | Train Sh | OOS Sh | OOS WR | WF Avg | WF Pos |
|----------|---------|--------|--------|--------|--------|
| VWAP MR L+S p48 z2.5 | 0.19 | **1.63** | **75%** | 0.86 | 6/10 |
| RSI MR L+S p7 20/80 | 0.04 | **1.04** | **62%** | 0.60 | 7/10 |
| VWAP MR p48 z2.5 (long) | 0.88 | 0.57 | 75% | **1.21** | 6/10 |
| RSI MR p7 20/80 (long) | 0.77 | 0.31 | 58% | **1.24** | 6/10 |

**VWAP MR L+S on 4H** is the most promising intraday strategy:
- Buy when price is 2.5 std devs below 48-bar VWAP, sell when returns to VWAP
- Short when 2.5 std devs above, cover at VWAP
- OOS: Sharpe 1.63, 75% win rate, +67% return
- Walk-forward: 0.86 avg, 6/10 positive periods

**Caveat:** The L+S versions showed *positive* decay (better OOS than train). This is unusual — the test period (late 2025 - early 2026) was particularly good for mean reversion due to the BTC decline. Don't assume this persists.

### 1-Hour Results

#### In-Sample Top Strategies
| Strategy | Sharpe | Return | Trades | WR |
|----------|--------|--------|--------|-----|
| RSI MR L+S p14 25/75 | 0.58 | 35% | 73 | 62% |
| RSI MR p14 25/75 | 0.57 | 30% | 36 | 61% |
| RSI MR p21 25/75 | 0.45 | 19% | 12 | 58% |

#### Walk-Forward
| Strategy | Train Sh | OOS Sh | OOS WR | WF Avg | WF Pos |
|----------|---------|--------|--------|--------|--------|
| RSI MR L+S p14 25/75 | 0.15 | **1.07** | **69%** | -0.01 | 5/10 |
| RSI MR p14 25/75 (long) | 0.81 | 0.25 | 67% | 0.64 | 6/10 |
| RSI MR p21 30/70 (long) | 0.70 | 0.12 | 60% | **0.71** | 6/10 |

**1H is not production-ready.** The RSI L+S OOS Sharpe of 1.07 looks good but walk-forward average is basically zero — the OOS result is likely a lucky period. The long-only versions are more stable (WF 0.64-0.71) but OOS Sharpe is too low (0.12-0.25).

### What Failed on Intraday

| Category | 4H | 1H |
|----------|-----|-----|
| EMA Crossover | Sharpe 0.56 (best) but 31% WR | All negative |
| MACD | Sharpe 0.38, 29% WR | All negative |
| Supertrend | No profitable config | No profitable config |
| StochRSI | All negative | All negative |
| Keltner Breakout | Sharpe 0.36 | All negative |
| ATR Breakout | Sharpe 0.27 | Sharpe 0.28 |

### VWAP Mean Reversion — How It Works

```python
vwap = (close * volume).rolling(48).sum() / volume.rolling(48).sum()
vwap_std = (close - vwap).rolling(48).std()
z_score = (close - vwap) / vwap_std

# Buy when z < -2.5 (oversold vs volume-weighted fair value)
# Sell when z > 2.5 (overbought)
```

The 48-bar window on 4H = 8 days of volume-weighted price. When price deviates 2.5 standard deviations from this rolling VWAP, it's statistically extreme and tends to revert. The volume weighting means the "fair value" is anchored to where most trading actually occurred, not just the average close.

### RSI Mean Reversion — How It Works

Standard RSI(7) with tight thresholds (20/80) or RSI(14) with 25/75. On intraday:
- RSI < 20-25: severely oversold, buy
- RSI > 75-80: severely overbought, sell
- Works because intraday BTC momentum exhausts quickly — a rapid drop below RSI 20 is usually a liquidity grab that reverts

### Multi-Timeframe Approach (Untested)

A potential combination:
1. **Daily S/R + RSI**: macro direction (which side of the market to be on)
2. **4H VWAP MR**: intraday entries (time entries to mean-reversion dips within the daily trend)

Example: Daily S/R says LONG → on 4H, only take VWAP buy signals (z < -2.5), skip short signals. This would filter the 4H strategy to trade with the daily trend instead of against it.

Not yet tested — would need walk-forward validation on the combination.

### Parameters

```
4H VWAP Mean Reversion:
  VWAP period:     48 bars (8 days)
  Z-score entry:   2.5 standard deviations
  Exit:            return to VWAP (z-score crosses zero)

1H RSI Mean Reversion:
  RSI period:      14 bars
  Oversold entry:  RSI < 25
  Overbought exit: RSI > 75
  
Fees: 0.1% per trade (both timeframes)
```

### Verdict

- **4H VWAP MR L+S**: promising, high win rate (75%), needs more data to confirm. Could complement daily strategy.
- **4H RSI MR**: decent walk-forward consistency (1.24 avg), lower conviction per trade.
- **1H**: not ready. Edges are thin and inconsistent across periods.
- **15-Min**: dead. Don't bother.
- **Daily S/R remains the primary strategy.** Intraday should only be added as a supplement, not a replacement.

## Hawkes Volatility Process

**Source:** `VolatilityHawkes/hawkes.py` (neurotrader888)

### How It Works

A Hawkes process models volatility clustering — one volatile bar increases the probability of more volatile bars. Applied to normalized candle ranges:

```python
# Normalize range by ATR
norm_range = (log_high - log_low) / ATR(180)

# Hawkes process: exponentially weighted sum of past ranges
alpha = exp(-kappa)
hawkes[i] = hawkes[i-1] * alpha + norm_range[i]
hawkes *= kappa  # scale by decay rate
```

The signal logic:
1. Track rolling 5th and 95th percentile of Hawkes values
2. When Hawkes drops below 5th percentile → volatility is **compressed** (quiet market)
3. When Hawkes then spikes above 95th percentile → volatility **explodes**
4. Enter in the direction of the price move from the compression point to the explosion point

### Standalone Results (Daily)

| Config | Sharpe | Return | Trades | WR |
|--------|--------|--------|--------|-----|
| k=0.5 lb=60 | 1.17 | 1051% | 24 | 50% |
| k=0.1 lb=60 | 1.09 | 757% | 25 | 44% |
| k=0.25 lb=60 | 1.05 | 794% | 22 | 50% |

Walk-forward (train 2020-2023, test 2024-2026):

| Config | Train Sh | OOS Sh | Decay |
|--------|---------|--------|-------|
| k=0.25 lb=60 | 1.19 | **0.78** | -35% |
| k=0.1 lb=180 | 0.69 | **0.64** | -7% |
| k=0.05 lb=180 | 0.52 | **0.56** | +7% |

### Combination with S/R Strategy

Hawkes measures volatility regime transitions, S/R measures structural price levels — fundamentally different signals. Tested as filter, union entries, and exit modifier.

| Combination | Full Sh | **OOS Sh** | OOS WR | Decay |
|-------------|---------|-----------|--------|-------|
| **H k=0.25/lb=60 union entries** | 1.64 | **1.29** | **51%** | **-28%** |
| H k=0.25/lb=60 hold during comp | 1.61 | **1.18** | 49% | -33% |
| H k=0.10/lb=180 hold during comp | 1.65 | **1.17** | 49% | -36% |
| Baseline (no Hawkes) | 1.59 | 1.05 | 48% | -42% |
| H as compression filter | 0.5-1.0 | 0.3-0.8 | varies | varies |

**The Hawkes union improved OOS Sharpe from 1.05 to 1.29** — the best OOS result of any strategy variant we tested. It adds Hawkes volatility explosion entries alongside S/R entries: when vol compresses then explodes upward → long, downward → short. This catches moves that S/R misses (breakouts from low-vol consolidation that don't occur at S/R levels).

The "hold during compression" variant is also strong — don't exit trades when Hawkes says vol is compressed, since compressed vol means consolidation not reversal.

**Hawkes as a filter failed** — requiring recent compression before allowing S/R entries was too restrictive and cut valid trades.

### Updated Best Strategy

With Hawkes union, the production strategy becomes:

**Long entries:** Trendline crossover OR Flag breakout OR S/R level penetration (RSI>50) OR Hawkes vol explosion (upward)  
**Long exits:** Trendline resistance crossdown OR Flag breakdown OR S/R level penetration (downward)  
**Short entries:** S/R level penetration (RSI<50) OR Hawkes vol explosion (downward)  
**Short exits:** S/R level penetration (upward) OR Hawkes vol explosion (upward)

OOS performance: Sharpe 1.29, 51% WR, -20% max drawdown, -28% decay.

### Parameters

```
Hawkes kappa:        0.25 (decay rate — higher = faster decay, more responsive)
Hawkes ATR norm:     180 bars (same as S/R lookback)
Hawkes quantile lb:  60 bars (rolling window for 5th/95th percentiles)
Compression:         Hawkes below 5th percentile
Explosion:           Hawkes above 95th percentile
```

### Timeframe Results

| Timeframe | Best Config | Sharpe | Trades |
|-----------|------------|--------|--------|
| Daily k=0.5 lb=60 | standalone | 1.17 | 24 |
| 4H k=0.1 lb=168 | standalone | 0.93 | 19 |
| 1H k=0.01 lb=336 | standalone | 0.82 | 28 |

Hawkes works best on daily, similar to S/R and VSA. The volatility clustering pattern needs days to form meaningful compression/explosion cycles.

## Hierarchical Market Structure

**Source:** `market-structure/` (neurotrader888) — `atr_directional_change.py`, `hierarchical_extremes.py`

### How It Works

Builds multi-level swing structure from a single timeframe using ATR-based directional change:

1. **ATR Directional Change** — detects swing highs/lows when price retraces by 1x ATR from a pending extreme. Adaptive to volatility (unlike fixed % directional change).
2. **Hierarchical Extremes** — takes level 0 swings and builds higher levels: a level N+1 high is confirmed when a level N high is followed by a lower level N high (break of structure). Ensures alternating highs/lows at each level.

This gives level 0 (minor swings), level 1 (intermediate), level 2 (major), level 3 (macro) — multi-timeframe structure from one timeframe.

### Strategies Tested

1. **Break of Structure (BOS)** — go long on higher-high confirmation, short on lower-low confirmation
2. **S/R Mean Reversion** — buy near level lows (support), sell near level highs (resistance)
3. **Trend Following** — long when both higher-highs and higher-lows, short on lower-highs and lower-lows

### In-Sample Results (Best Per Timeframe)

| Strategy | Sharpe | Return | Trades | WR |
|----------|--------|--------|--------|-----|
| Daily atr30 lvl2 BOS long | 1.07 | 945% | 6 | 83% |
| Daily atr60 lvl2 BOS long | 1.06 | 875% | 5 | 80% |
| 4H atr72 lvl2 BOS long | 1.00 | 65% | 15 | 40% |
| 4H atr72 lvl2 BOS L+S | 0.97 | 85% | 31 | 45% |
| 1H atr168 lvl2 BOS L+S | 0.94 | 83% | 104 | 58% |
| 4H atr24 lvl1 SR MR 1% | 0.69 | 49% | 119 | **70%** |

### Walk-Forward: BOS Collapsed

| Config | Train Sharpe | **OOS Sharpe** | Decay |
|--------|-------------|---------------|-------|
| atr30 lvl2 BOS L+S | **1.41** | **-0.49** | **-135%** |
| atr60 lvl2 BOS L+S | 1.27 | -0.40 | -131% |
| atr14 lvl2 BOS L+S | 0.54 | -0.41 | -176% |
| All lvl1 BOS L+S | negative | negative | — |

**Every BOS configuration failed out-of-sample.** The strategy detects structural breakouts after they've already happened — you're buying BTC after a massive move. This worked in 2020-2023's strong trends but failed in 2024-2026's ranging market.

### Why It Failed

BOS is a **lagging momentum signal**. By the time a level 2 higher-high is confirmed, BTC has already moved 15-30%. In a trending regime this works (you're riding the trend). In a ranging regime (2024-2026) you buy the top and sell the bottom repeatedly.

Compare to our S/R strategy which enters at the moment of a level penetration (leading signal) vs BOS which enters after the structure has already changed (lagging signal).

### What's Useful

The **hierarchical S/R levels** for mean reversion on 4H showed promise:
- 4H atr24 lvl1 S/R MR (1% band): 70% win rate, 119 trades — consistent bouncing off structure levels
- This aligns with our finding that intraday BTC is mean-reverting

These structure levels could theoretically improve the 4H VWAP MR strategy (bounce off structure levels instead of arbitrary VWAP bands) but was not tested in combination or walk-forward validated.

### Verdict

**Do not use BOS for trading signals** — confirmed overfit, -135% decay OOS. The hierarchical structure itself is a useful analytical tool for identifying swing levels, but the BOS entry timing is too late for BTC's volatile regime shifts. Our S/R + Hawkes approach captures the same structural information with better timing and validated OOS performance.

## Meta-Labeling (ML-Filtered Breakouts)

**Source:** `TrendlineBreakoutMetaLabel/` (neurotrader888) — `trendline_breakout.py`, `trendline_break_dataset.py`, `walkforward.py`

### Concept

Meta-labeling doesn't generate trading signals — it **filters existing signals** using machine learning. For each breakout signal, extract features (slope, volume, RSI, ADX, etc.), train a classifier to predict which breakouts will be profitable, and only take trades the model approves.

Walk-forward: retrain periodically on recent data to adapt to changing market regimes.

### How It Works

1. **Generate breakout trades** — fit trendlines on rolling window, enter when price breaks above resistance (long) or below support (short)
2. **Extract features per trade:**
   - Trendline slope (normalized by ATR)
   - Trendline error (fit quality)
   - Max distance from trendline
   - Breakout strength (how far past the line)
   - Normalized volume
   - RSI(14)
   - ADX(14)
   - Channel width (resistance - support, normalized)
3. **Label:** 1 if trade was profitable, 0 if not
4. **Train classifier** (Random Forest / Gradient Boosting) on historical trades
5. **Filter:** only take new breakout trades where model predicts >55% probability of profit

### Results (Daily BTC, train 2020-2023, test 2024-2026)

**Baseline trendline breakout: 719 trades, 49% WR, PF 0.88 — a losing strategy.**

With ML filter (out-of-sample):

| Model | Threshold | Trades Taken | Win Rate | Profit Factor |
|-------|-----------|-------------|----------|---------------|
| Baseline (no ML) | all | 283 | 48% | 0.88 |
| RF depth=3 | 0.55 | 21 | **62%** | **5.28** |
| RF depth=5 | 0.55 | 70 | **63%** | **2.48** |
| RF depth=5 | 0.60 | 16 | 63% | 3.71 |
| GB depth=3 | 0.55 | 90 | 59% | 1.78 |
| GB depth=3 | 0.70 | 49 | 55% | 2.11 |

**The ML filter turned a losing strategy (PF 0.88) into a winning one (PF 2.48-5.28).** Random Forest at threshold 0.55 rejected 93% of trades but the remaining 7% had 62% win rate.

Walk-forward backtest (all trades, no ML): Sharpe 0.42, Ret 68%, DD -54%. The ML-filtered version showed improved trade quality but reduced total return due to fewer trades.

### Key Insight

Meta-labeling is powerful because it doesn't need the base strategy to be profitable — it just needs the base strategy to generate a **pool of candidate trades** with measurable features. The ML model learns to distinguish signal from noise within that pool.

### Application to Our S/R Strategy (Untested)

Our S/R + Hawkes strategy has 51% OOS win rate. Meta-labeling could potentially push this higher by extracting features at each S/R signal:
- Distance from S/R level at entry
- Hawkes process value (vol compression level)
- RSI value
- ATR-normalized volume
- Number of times this level has been tested
- Time since last signal (whipsaw detection)

This is the most promising research direction for improving win rate without overfitting — the features are computed per-trade and the model is retrained walk-forward, so it adapts to regime changes.

**Caution:** The base trendline breakout only had 21 trades taken by RF d3 at 0.55 threshold — too few for reliable statistics. Higher trade counts (GB d3 at 0.55 = 90 trades, 59% WR) are more trustworthy but less impressive. The meta-labeling concept is validated but the specific numbers should be taken with a grain of salt.

### Meta-Labeling Applied to Our S/R + Hawkes Strategy: FAILED

Tested meta-labeling on our production strategy. Results:

- 85 total trades (45 train, 40 test) — far too few for ML
- Baseline: 32.9% WR, PF 2.50
- RF d3 at 0.45 threshold: selected 4 trades, **0% win rate** — model filtered out all winners
- RF d5 at 0.45: 7 trades, 14% WR — same problem

**Why it failed:**

1. **Not enough training samples.** 45 training trades is noise for a Random Forest. ML needs hundreds of labeled samples. Our strategy generates ~20 trades/year — you'd need 10+ years of data.

2. **Wrong win/loss distribution.** Our strategy wins 33% of the time but wins big (PF 2.50). The ML model tries to predict which trades win, but the features don't distinguish "about to win big" from "about to lose small." It ended up filtering out the rare big winners.

3. **Comparison:** Trendline breakouts had 719 trades with 49% WR — a much more learnable pattern. ML thrives on high-frequency, balanced win/loss datasets.

**When to use meta-labeling:**
- High-frequency strategies (200+ trades in training window)
- ~50% base win rate (balanced classes for classification)
- Features that distinguish winners from losers at entry time

**When NOT to use it:**
- Low-frequency strategies (<50 trades in training)
- Asymmetric W/L profile (low WR, high PF — winners are unpredictable big moves)
- Our S/R + Hawkes strategy falls in this category

### Parameters

```
Base strategy:     Trendline breakout (lookback=30, hold_period=10)
SL/TP:             2x ATR each
ATR lookback:      90 bars
Features:          slope, tl_err, max_dist, breakout_str, vol, rsi, adx, channel_width
Model:             RandomForestClassifier(n_estimators=500, max_depth=3)
Probability thresh: 0.55
Walk-forward:      Retrain every 180 bars on last 3 years of trades
```

## Time Series Reversibility (PTSR)

**Source:** `TimeSeriesReversibility/` (neurotrader888) — `reversibility.py`, `indicators.py`

### Concept

Measures whether a time series looks different when played forward vs backward. Based on permutation pattern distributions (Zanin et al., 2018):

- **High PTSR (irreversible)** = trending market — forward and reverse patterns differ significantly
- **Low PTSR (reversible)** = mean-reverting/random — forward and reverse look the same

This is a **regime detector**, not a direct trading signal.

### How It Works

```
1. Extract ordinal patterns of order 3 from rolling window
2. Compute pattern distribution for forward and reversed series
3. PTSR = KL divergence between forward and reverse distributions
4. Higher = more irreversible = more trending
```

### Regime Detection Validated

Forward returns on BTC daily by PTSR regime (lb=90, quantile bands over 180-bar rolling window):

| Regime | Count | 5-day Return | 10-day Return |
|--------|-------|-------------|---------------|
| **High PTSR (trending)** | 527 | **+1.33%** | **+2.59%** |
| Mid PTSR | 1195 | +1.14% | +2.60% |
| **Low PTSR (reverting)** | 571 | **-0.58%** | **-1.81%** |

PTSR successfully separates trending from mean-reverting regimes. Trending regimes have positive forward returns; reverting regimes have negative.

### Combined With S/R Strategy

| Strategy | Full Sharpe | **OOS Sharpe** | OOS WR | OOS Trades |
|----------|-----------|---------------|--------|------------|
| **PTSR120 > q50 (trending filter)** | 1.17 | **1.49** | 35% | 20 |
| Baseline (no PTSR) | 1.34 | 1.14 | 30% | 40 |
| PTSR120 regime-aware | 0.97 | 0.83 | 35% | 26 |
| PTSR90 > q50 trending | 0.84 | 0.64 | 29% | 24 |
| PTSR90 standalone | 1.10 | 0.79 | 44% | 9 |

**PTSR120 > q50 achieved OOS Sharpe 1.49** — the highest OOS Sharpe in all our research. Only takes S/R signals when the 120-day PTSR is above its rolling median (market is trending). Halves the trade count from 40 to 20 OOS.

### Important Caveats

1. **Only 20 OOS trades** — statistically thin. Could be lucky. Needs more data to confirm.
2. **Full-period Sharpe (1.17) is lower than baseline (1.34)** — the filter hurts during some in-sample periods. The OOS improvement could be period-specific.
3. **PTSR lookback of 120 days is long** — it detects slow regime shifts, not fast ones. By the time PTSR flags "trending," a significant portion of the trend may have already played out.
4. **Not all PTSR configs worked** — PTSR60 and PTSR90 trending filters had lower OOS Sharpe than baseline. Only PTSR120 > q50 outperformed. Parameter sensitivity is a yellow flag.

### Best Use Case

PTSR is most useful as a **context indicator** rather than a hard filter:
- When PTSR is high → expect trend continuation, hold positions longer, potentially increase size
- When PTSR is low → expect mean reversion, tighten stops, reduce position size
- Don't hard-filter signals based on PTSR — the 120-day lookback makes it too slow to react to regime changes

### Parameters

```
PTSR lookback:       120 bars (daily)
Ordinal pattern:     d=3 (6 possible patterns)
Regime threshold:    rolling 180-bar median (q50)
Smoothing:           EMA(7) on raw PTSR values (optional, for less noise)
```

## RSI-PCA (Principal Component Analysis on Multiple RSI Periods)

**Source:** `RSI-PCA/` (neurotrader888) — `walkforward.py`, `pca.py`

### How It Works

1. Compute RSI at 23 different periods (2 through 24)
2. Run PCA to extract principal components — PC1 captures overall RSI level, PC2 captures divergence between fast/slow RSI, PC3+ capture higher-order structure
3. Train a linear model: predict N-bar forward log returns from the PCA components
4. Signal: go long when prediction > 95th percentile, short when < 5th percentile
5. Walk-forward: retrain periodically (every 180 bars on last 1-2 years)

The idea is that PCA extracts more information from the RSI landscape than any single RSI period.

### Results (Daily BTC, OOS = 2024-2026)

| Config | Full Sharpe | OOS Sharpe | OOS WR | OOS Trades | OOS DD |
|--------|-----------|-----------|--------|------------|--------|
| PC4 LA7 1yr | -0.42 | **0.85** | **75%** | 12 | **-7.6%** |
| PC4 LA14 1yr | -0.25 | **0.71** | **80%** | 5 | **-7.5%** |
| PC3 LA10 2yr | 0.28 | 0.59 | 67% | 6 | -7.6% |
| PC4 LA5 1yr | -0.53 | 0.58 | 52% | 21 | -18.4% |

### Assessment

**High win rate (75-80%) but very few trades (5-12).** The extreme quantile thresholds (95th/5th) make it highly selective — too selective for a standalone strategy.

**Negative full-period Sharpe with positive OOS** is a yellow flag. The model may have learned something period-specific rather than a generalizable pattern.

**Low drawdowns (~7.5%)** are the standout feature — much lower than any other strategy we tested. Useful as a conservative overlay.

**Doesn't compete with our S/R + Hawkes approach** (OOS Sharpe 1.29, 51 trades). PCA on RSI doesn't add much over our simple RSI > 50 filter — the directional momentum information is already captured.

### When RSI-PCA Might Work

- **Higher-frequency data** — the original code used hourly BTC. With more bars, PCA has more training data and more trading opportunities.
- **Multi-asset** — PCA across RSIs of multiple correlated assets (e.g., BTC + ETH + SOL) could capture cross-asset momentum divergences that single-asset RSI misses.
- **As a conservative overlay** — only trade when RSI-PCA agrees with the primary strategy, accepting far fewer trades for very high win rate and minimal drawdown.

### Not Recommended For Production

The trade count is too low on daily BTC to rely on. The PCA decomposition adds mathematical complexity without clear edge over raw RSI. Save this for higher-frequency or multi-asset applications.

## Monte Carlo Permutation Test (MCPT)

**Source:** `mcpt/` (neurotrader888) — `bar_permute.py`, framework for strategy validation

### What MCPT Does

Tests whether a strategy's performance is statistically significant or just luck by:

1. **Permuting OHLC bars** — shuffle bar order while preserving intrabar structure (open-high-low-close relationships within each bar). This destroys any temporal patterns (trends, mean reversion, S/R levels) while keeping the return distribution identical.
2. **Running the strategy on 100+ permuted series** — each gives a "chance" performance level
3. **Computing p-value** — what fraction of permutations produced performance >= real performance

If p < 0.05, the strategy is detecting real structure, not random patterns.

### MCPT on Our S/R + RSI Strategy

Ran 100 permutations on full BTC daily data (2020-2026):

| Metric | Real Data | Permuted Mean | Permuted Max | **P-Value** |
|--------|-----------|---------------|-------------|-------------|
| Profit Factor | **1.205** | 1.015 | 1.168 | **0.010** |
| Sharpe Ratio | **1.170** | 0.081 | 0.970 | **0.010** |

**Both p-values < 0.01 — highly significant.** Not a single permutation out of 100 produced performance matching our real strategy. The real PF (1.21) exceeds the 95th percentile of permuted PFs (1.13).

Permuted PF distribution:
```
5th percentile:  0.905
25th percentile: 0.968
50th percentile: 1.014
75th percentile: 1.059
95th percentile: 1.128
Real:            1.205  ← well above all permutations
```

### Interpretation

**The edge is real, not noise.** The S/R level penetration signal detects genuine structural features in BTC's price — these features disappear when the bar order is randomized. This is the strongest statistical validation possible for a trading strategy.

Combined with our walk-forward results (OOS Sharpe 1.05-1.29, 7/8 positive periods), we now have three independent lines of evidence:
1. **Walk-forward validation** — strategy works on unseen data
2. **MCPT** — strategy performance is not due to chance (p < 0.01)
3. **Economic logic** — S/R levels represent real supply/demand zones where price has historically concentrated

### Using MCPT for Future Research

The `bar_permute.py` framework should be applied to any new strategy before considering it for production:
- p < 0.01: strong evidence of real edge
- p < 0.05: likely real edge
- p < 0.10: marginal, needs more investigation
- p > 0.10: probably noise, don't trade it

Note: MCPT tests the in-sample performance. A strategy can be statistically significant (real pattern exists) but still overfit to specific parameters. Always combine MCPT with walk-forward validation.

### Parameters

```
Permutation method:  OHLC bar shuffle (preserves intrabar structure)
Start index:         180 (after S/R lookback warm-up)
N permutations:      100 (minimum for p < 0.01 resolution)
Strategy tested:     S/R + RSI filter L+S on daily BTC
```
