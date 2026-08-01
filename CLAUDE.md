# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vectorbt is a Python library for backtesting and analyzing trading strategies at scale. It uses NumPy/Pandas with Numba JIT compilation for performance-critical paths. Licensed under Apache 2.0 with Commons Clause.

## Common Commands

```bash
# Install dependencies (uses uv)
uv sync --extra cov

# Run all tests
uv run pytest tests/

# Run a specific test file
uv run pytest tests/test_portfolio.py

# Run a specific test by name
uv run pytest tests/test_base.py -k "test_name"

# Run with coverage
uv run pytest tests/ --cov=vectorbt

# Type checking
uv run mypy vectorbt/

# Build docs (from docs/ directory)
cd docs && python generate_api.py && mkdocs build --strict
```

## Architecture

### Core Design Patterns

**Pandas Accessor Pattern:** The library extends pandas via `.vbt` accessors registered on Series/DataFrames. Each module (generic, signals, returns, etc.) registers its own accessor. Entry point: `import vectorbt as vbt`.

**Numba JIT Compilation:** Every module splits logic into two layers:
- `nb.py` — Numba `@njit`-compiled functions for hot loops (e.g., `portfolio/nb.py` at 270KB is the core backtesting engine)
- Python-level classes/accessors that call into the `nb.py` functions

When editing `nb.py` files, all functions must be Numba-compatible (no Python objects, limited NumPy API).

**Factory Pattern:** `indicators/factory.py` and `signals/factory.py` dynamically generate indicator/signal classes. Custom indicators are built by defining a Numba kernel and passing it through the factory.

**Configuration:** Global settings via `vbt.settings` (defined in `_settings.py`). Supports nested dot-access and dict-style access.

### Module Dependency Flow

```
data (YFinance, CCXT, Alpaca) → base (ArrayWrapper, indexing)
    → generic (array ops, plotting) → indicators / signals
    → portfolio (backtesting engine) → records (trades, orders)
    → returns (metrics) → labels (ML labels)
```

### Key Modules

- **base/** — `ArrayWrapper`, `ColumnGrouper`, indexing/reshaping primitives that all other modules build on
- **portfolio/** — Core backtesting: `Portfolio.from_signals()`, `Portfolio.from_orders()`, order/trade records. The `nb.py` here is the performance-critical simulation loop
- **indicators/** — Factory-generated technical indicators (MA, BBANDS, RSI, etc.)
- **records/** — Typed record arrays (`MappedArray`, column mapping) for storing variable-length per-column data (trades, drawdowns, etc.)
- **generic/** — Generic vectorized operations, splitters, ranges, drawdown analysis
- **utils/** — Config system, decorators, datetime helpers, array utilities, checks

### Test Structure

Tests are in `tests/` with one file per module. Test files are large (e.g., `test_portfolio.py` is 340KB) and use hash-based comparison via `tests/utils.py` helpers (`isclose`, `record_arrays_close`). Tests compare against expected hash values rather than literal expected outputs.

## Important Constraints

- Python >=3.10, pandas >=2.0 <3.0, numba >=0.60
- Numba functions in `nb.py` files cannot use Python objects or arbitrary libraries — only NumPy primitives and Numba-supported operations
- The project uses `uv` for dependency management (no requirements.txt — dependencies are in pyproject.toml)

## Strategy Research & Backtesting Guide

Hard-won lessons from extensive BTC strategy research. See `RESEARCH.md` for the full study.

### The #1 Rule: Walk-Forward Before You Celebrate

Never trust in-sample results. Every strategy looks great on the data it was optimized on.

**Required validation process:**
1. **Train/test split** — optimize parameters on train (e.g., 2020-2023), evaluate blind on test (2024+). Expect 30-40% Sharpe decay. If decay exceeds 50%, the strategy is overfit.
2. **Rolling walk-forward** — test on 6-month rolling windows (at least 6-8 windows). A real edge shows positive Sharpe in 75%+ of windows. Below 60% is noise.
3. **Only then** look at full-sample metrics.

### Overfitting Red Flags

- In-sample Sharpe > 2.0 on crypto — almost certainly overfit
- Returns that look "too good" (>3000% on BTC) — the parameters are curved-fitted to known price moves
- Filters that fix one specific drawdown period — they will fail on the next one (we proved this: a whipsaw filter that cut 2021 drawdown from -53% to -21% in-sample had -89% Sharpe decay out-of-sample)
- Strategies with many tunable parameters — each parameter is an overfitting opportunity
- Win rate jumps dramatically after adding a filter — check OOS before trusting it

### What Survives Walk-Forward (Lessons Learned)

**Structural signals beat parameter-fitted signals.** S/R levels from KDE market profiles survived all walk-forward tests because they represent real price zones where supply/demand accumulated. Trendline slope thresholds (-0.001, 0.003) were tuned to specific BTC behavior and collapsed OOS.

**Fewer parameters = more robust.** Our best OOS strategy (S/R L+S) has essentially zero tunable parameters beyond the KDE lookback (180). The more parameters you optimize, the more you're fitting noise.

**Asymmetric designs work.** Using different signal systems for longs vs shorts outperformed symmetric approaches. BTC's upward bias means long signals can be looser (cast a wide net) while short signals must be selective.

**Simple filters that make logical sense hold up.** RSI > 50 for longs (trade with momentum) survived OOS. Complex filters (Hold5+ATR<6%) that were engineered to avoid a specific past event did not.

### Backtesting with vectorbt — Practical Tips

**Frequency issues:** BTC daily data from yfinance has daily frequency and works directly. S&P 500 has business day frequency which vectorbt can't infer — pass `freq="1D"` explicitly. Intraday data often has `None` frequency — always set `freq` manually.

**Long + Short:** Use `short_entries` and `short_exits` parameters in `Portfolio.from_signals()`. Don't use `direction="Both"` with a single entry/exit pair — it's less controllable.

```python
pf = vbt.Portfolio.from_signals(
    close=price,
    entries=long_entries, exits=long_exits,
    short_entries=short_entries, short_exits=short_exits,
    init_cash=10_000, fees=0.001, freq="1D"
)
```

**SL/TP:** Pass `sl_stop`, `tp_stop` as arrays for per-bar dynamic values (e.g., ATR-based stops). Also pass `open`, `high`, `low` for accurate intrabar stop execution. `sl_trail=True` enables trailing stops. On BTC: fixed TP consistently hurt (caps winners), SL only helped at tight levels (~3% or 0.75x ATR).

**Key metrics to check:**
- `pf.sharpe_ratio()`, `pf.sortino_ratio()`, `pf.max_drawdown()`
- `pf.trades.count()`, `pf.trades.win_rate()`, `pf.trades.profit_factor()`
- `pf.trades.records_readable` for individual trade inspection
- `pf.value()` for equity curve

### What Doesn't Work (Don't Waste Time)

- **Intraday S/R on BTC** — S/R levels are macro structures. On 4h/1h/15m, price crosses them dozens of times per day as noise. Fees alone destroy the account at 1h frequency.
- **Trendline slopes as the sole signal** — overfit to specific regimes, -78% OOS decay
- **Take profit on BTC** — BTC trends are long and fat-tailed. TP cuts the tail.
- **Trailing stops on BTC daily** — intraday volatility (3-5% ATR) triggers them before trends develop
- **Confirmation bars (require N bars above level)** — reduces trades too aggressively without proportional quality improvement
- **ADX trend filter** — didn't reliably distinguish trending from choppy on BTC
- **Applying BTC strategies to equities** — tested on S&P 500, no strategy beat buy & hold

### Leverage

Kelly Criterion for our best strategy: Full Kelly = 3.4x, Half Kelly = 1.7x.

- **1.5x**: recommended — worst day -17%, max drawdown -27%
- **2.0x**: aggressive — worst day -22%, max drawdown -35%
- **3.0x**: maximum viable — returns peak here then decline due to drawdown compounding
- **>5x**: returns collapse. **>7x**: liquidation on a single bad day

### Pattern Library

The sibling repo `../TechnicalAnalysisAutomation/` (neurotrader888) contains pattern detectors that integrate with vectorbt:

| Module | Signal Type | BTC Effectiveness |
|--------|------------|-------------------|
| `mp_support_resist.py` | KDE S/R level penetration | **Core edge** — survived all OOS tests |
| `trendline_automation.py` | Support/resistance slope crossover | Long entries only, overfit as sole signal |
| `flags_pennants.py` | Bull/bear flag breakout | Rare but high quality (~15 signals in 6 years) |
| `head_shoulders.py` | H&S / IH&S reversal | Useful as exit signal, not entry |
| `harmonic_patterns.py` | XABCD Fibonacci patterns | Marginal on daily BTC |
| `directional_change.py` | Zigzag turning points | Too noisy standalone |
| `pip_pattern_miner.py` | ML clustering of price shapes | Requires pyclustering, not tested in walk-forward |

## Related Project: autoresearch

The sibling repo at `../autoresearch/` is used alongside vectorbt. It is an autonomous AI research system (based on Andrej Karpathy's concept) where an LLM agent iteratively modifies a training script, runs 5-minute GPU experiments, and keeps or discards changes based on a target metric.

### Key Files

- **`train.py`** — The only file the agent edits. Contains a GPT model (RoPE, Flash Attention 3, sliding window, MuonAdamW optimizer). ~50-100M params.
- **`prepare.py`** — Read-only. Data download (climbmix-400b from HuggingFace), BPE tokenizer training (8192 vocab), BOS-aligned packing dataloader, and `evaluate_bpb()` evaluation function.
- **`program.md`** — Instructions for the AI agent (edited by humans to steer research direction).
- **`analysis.ipynb`** — Analyzes `results.tsv` (experiment log with commit/val_bpb/status/description). Plots progress and ranks improvements.

### How It Works

Each experiment: modify `train.py` → train for exactly 5 minutes → evaluate `val_bpb` (bits per byte) → keep/discard. ~12 experiments/hour. Requires single NVIDIA GPU (tested on H100).

### Integration with vectorbt

The two projects can form an end-to-end ML trading pipeline:
1. **Data**: vectorbt downloads OHLCV data via its data module (YFinance, CCXT, Alpaca)
2. **Model**: autoresearch trains/optimizes a neural model on market data
3. **Signals**: Model predictions feed into vectorbt signal generation
4. **Backtesting**: `vbt.Portfolio.from_signals()` evaluates strategy performance
5. **Metric**: Replace `val_bpb` with trading metrics (Sharpe ratio, max drawdown) to autonomously optimize strategies
