<div align="center">

# ALPHA-PRIME

### Autonomous AI-Powered Intraday Trading System for Indian Markets

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-blueviolet.svg)](https://langchain-ai.github.io/langgraph/)
[![OpenAI GPT-4o](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)](https://openai.com/)
[![Groww API](https://img.shields.io/badge/Broker-Groww%20API-orange.svg)](https://groww.in/trade-api)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)

**A fully autonomous AI-powered intraday trading system that scans the NSE market, analyzes stocks using 10+ technical indicators, runs a Bull/Bear debate between opposing AI agents, checks real-time news sentiment with GPT-4o-mini, makes risk-managed decisions, and executes live trades through the Groww broker API -- all without human intervention.**

**[>> Live Demo](https://alpha-prime.streamlit.app)** -- read-only dashboard over real paper-trading history

[Live Demo](#live-demo) |
[Results](#results-what-actually-happened) |
[Getting Started](#getting-started) |
[Architecture](#architecture) |
[Pipeline](#the-multi-agent-pipeline) |
[Features](#features) |
[Configuration](#configuration)

---

</div>

<br>

## Live Demo

**[alpha-prime.streamlit.app](https://alpha-prime.streamlit.app)**

The hosted dashboard is a read-only showcase. It runs against a committed
snapshot of a real paper-trading session -- 87 trades, 689 signals and 18,000
agent-reasoning log entries -- plus backtest output. It holds no API keys,
places no orders and makes no LLM calls, so it costs nothing to run and cannot
be made to do anything by a visitor. Execution controls are not merely disabled
in the UI; in demo mode the broker, scheduler and LLM modules are never even
imported (`alpha_prime/ui/dashboard.py`), and credential writes raise at the
settings layer (`alpha_prime/core/config.py`).

The app sleeps after inactivity and takes ~30 seconds to wake on the first hit.

<br>

## Results: what actually happened

Being straight about this, because the interesting engineering is in the failure.

**The paper-trading run lost money: -Rs 50,367 over 37 closed positions, a 35%
win rate.** The signals were not the main problem. Position sizing was.

| | |
|---|---|
| Total realized P&L | **-Rs 50,367** |
| Closed positions | 37 |
| Win rate | 35.1% |
| Average win | Rs 35 |
| Average loss | **Rs 4,620** |
| Worst two trades | ZYDUSLIFE -Rs 32,369, DIVISLAB -Rs 17,381 |

Two trades produced **98.8% of all losses**. Reconstructing the P&L showed why:
`max_position_size_pct` was interpreted against *margin buying power* rather than
real capital, so a single name could absorb Rs 12.5L of a Rs 10L account. Worse,
that limit and the daily-loss limit existed **only as instructions in the risk
manager's LLM prompt**. Nothing in the code enforced them.

That is the central lesson of this project, and it generalises past trading:

> **An LLM cannot be trusted to enforce a numeric constraint. If a limit matters,
> it belongs in code.**

The same failure appeared earlier in signal generation: agents reported a signal
strength of `1.0` regardless of a prompt that explicitly forbade it. The fix was
the same shape -- `save_signal` now re-runs the deterministic analysis engine and
records the *engine's* strength, discarding whatever the model claimed.

### What changed

| Problem | Fix |
|---|---|
| Size limits lived in a prompt | `_enforce_risk_limits()` in `agents/tools.py` clamps quantity in code: risk-per-trade against real capital, notional cap, max open positions |
| No daily-loss circuit breaker | Enforced in `place_trade` and in the scheduler cycle |
| Stops were in-process only | Breakeven arming plus trailing logic in `core/risk.py`; positions carried past their day are squared off |
| `pnl` never populated | Per-SELL realized P&L computed on fill, for both paper and live |
| Repeated losses on one name | 3-day cooldown per symbol after a loss, checked in two places |

### Does it make money?

**Not yet, and the backtest is honest about why.** An event-driven, no-lookahead
backtest over 60 days of 5-minute bars on 12 NIFTY-50 names, with Indian intraday
costs modelled (brokerage, STT, exchange and SEBI fees, stamp duty, GST):

| | |
|---|---|
| Trades | 239 |
| Gross P&L | +Rs 38,949 |
| Transaction costs | **-Rs 28,070** |
| **Net P&L** | **+Rs 10,879 (+1.09%)** |
| Profit factor | 1.10 |
| Max drawdown | 2.08% |

**Costs consume roughly 72% of gross profit.** With an average win near 0.5%,
each round trip must clear about Rs 117 in fees before earning anything. That,
not signal quality or risk control, is the binding constraint.

A parameter sweep found wider targets win consistently (2.5% > 1.8% > 1.2%),
because a larger move dwarfs the fixed round-trip cost, and `DEFAULT_TARGET_PCT`
was raised accordingly. But that sweep scored parameters on the same window it
selected them from. Re-running those settings on a later 60-day window returned
**+1.09% against the +3.32% measured in-sample** -- most of the apparent edge did
not survive going out of sample. Walk-forward validation and slippage stress are
the honest next steps, and neither is done yet.

The position-sizing and daily-loss controls are unambiguously correct, because
they remove the catastrophic-loss tail. They do not by themselves make the
system profitable.

<br>

## How It Works

Alpha-Prime uses a **7-agent AI pipeline** built on LangGraph where each agent is a specialized GPT-4o-mini instance with access to real-time market tools. Every 15 minutes during market hours, the pipeline runs automatically:

```
Market Opens (9:15 AM)
       |
       v
  +---------+    +-----------+    +----------------+    +--------------+    +-----------+    +------------------+
  | SCANNER |--->| ANALYST   |--->| BULL/BEAR      |--->| RISK MANAGER |--->| EXECUTOR  |--->| PORTFOLIO REVIEW |
  |         |    |           |    | DEBATE         |    |              |    |           |    |                  |
  | Scans   |    | Deep TA   |    | Bull argues    |    | Position     |    | Places    |    | Tracks P&L,      |
  | NIFTY50 |    | + News    |    | FOR buying.    |    | sizing &     |    | live      |    | Sharpe ratio,    |
  | stocks  |    | Sentiment |    | Bear argues    |    | approval     |    | orders    |    | max drawdown,    |
  |         |    | Entry/SL/ |    | AGAINST.       |    |              |    | on Groww  |    | recommendations  |
  |         |    | Target    |    | Judge decides. |    |              |    |           |    |                  |
  +---------+    +-----------+    +----------------+    +--------------+    +-----------+    +------------------+
       |                                                                         |
       | Every 15 min                                                            | Every 2 min
       | during market hours                                                     | Position Monitor
       |                                                                         | checks SL/Target
       v                                                                         v
  Auto Square-Off at 3:15 PM                                            Trailing Stop-Loss
  (sells all positions)                                                 (auto-sells on reversal)
```

<br>

## The Multi-Agent Pipeline

The core of Alpha-Prime is a **LangGraph state machine** where 7 AI agents work in sequence. Each agent has specialized tools and a focused mission.

### Pipeline Architecture

```
                          +----------------------------------------------+
                          |           TradingState (shared)               |
                          |                                              |
                          |  scanner_output  | analyst_output            |
                          |  debate_output   | risk_output               |
                          |  execution_output| portfolio_output          |
                          |  current_phase   | trade_decisions | error   |
                          +----------------------------------------------+
                                           |
                     +---------------------+---------------------+
                     |                     |                     |
                     v                     v                     v
              LangGraph StateGraph    OpenAI GPT-4o-mini   15 LangChain Tools
              (orchestration)         (reasoning +         (market actions +
                                       sentiment)          news sentiment)

  =============================================================================

  PHASE 1                 PHASE 2                 PHASE 2.5
  MARKET SCANNER          TECHNICAL ANALYST        BULL/BEAR DEBATE
  +----------------+      +------------------+     +------------------+
  |                |      |                  |     |                  |
  | Tools:         |      | Tools:           |     | Bull Agent:      |
  | - check_market | ---> | - run_intraday   | --> |   argues FOR     |
  | - get_prices   |      |   _analysis      |     |   buying         |
  | - run_intraday |      | - check_news     |     |                  |
  |   _analysis    |      |   _sentiment     |     | Bear Agent:      |
  | - check_news   |      | - save_signal    |     |   argues AGAINST |
  |   _sentiment   |      |                  |     |   buying         |
  |                |      | 6 hard rules:    |     |                  |
  | Scans NIFTY 50 |      | a) signal=BUY    |     | Judge Agent:     |
  | stocks, checks |      | b) strength>=0.7 |     |   GO / NO-GO     |
  | news sentiment |      | c) trend=BULLISH |     |   decision       |
  | Picks 0-2 BUY  |      | d) 3+ indicators |     |                  |
  | candidates     |      | e) RSI 40-65     |     |                  |
  |                |      | f) news != NEG   |     |                  |
  +----------------+      +------------------+     +------------------+
                                                          |
                                                          v
  PHASE 3                 PHASE 4                  PHASE 5
  RISK MANAGER            EXECUTION AGENT          PORTFOLIO MANAGER
  +----------------+      +------------------+     +------------------+
  |                |      |                  |     |                  |
  | Validates:     |      | For each         |     | Reviews:         |
  | - Daily P&L    | ---> |   APPROVED trade:| --> | - All positions  |
  |   limits       |      | - Verify price   |     | - Day P&L        |
  | - Signal >= 0.65      | - Execute BUY    |     | - Sharpe ratio   |
  | - R:R >= 1:1.5 |      | - Confirm fill   |     | - Max drawdown   |
  | - Position size|      |                  |     | - Recommendations|
  | - Margin aware |      | Uses Groww API   |     |                  |
  |                |      | (MIS/Intraday)   |     |                  |
  | APPROVE/REJECT |      |                  |     |                  |
  +----------------+      +------------------+     +------------------+
```

### Agent Details

| # | Agent | Role | Key Tools | Output |
|---|-------|------|-----------|--------|
| 1 | **Market Scanner** | Scans NIFTY 50 for momentum setups | `run_intraday_analysis`, `check_news_sentiment` | Top 0-2 BUY candidates |
| 2 | **Technical Analyst** | Deep analysis with 10+ indicators + news | `run_intraday_analysis`, `check_news_sentiment`, `save_signal` | Entry, SL, Target + sentiment check |
| 3 | **Bull Agent** | Argues FOR buying each stock | `run_intraday_analysis` | Bullish case with data |
| 4 | **Bear Agent** | Argues AGAINST buying each stock | `run_intraday_analysis` | Bearish case with risks |
| 5 | **Debate Judge** | Evaluates both sides, makes GO/NO-GO | None (pure reasoning) | Final verdict per stock |
| 6 | **Risk Manager** | Position sizing, margin-aware approval | `get_portfolio`, `get_current_positions` | APPROVED trades with quantity |
| 7 | **Execution Agent** | Places live orders on Groww | `place_trade`, `get_stock_price` | Order confirmations |
| 8 | **Portfolio Manager** | Reviews session, tracks performance | `get_portfolio`, `get_recent_trades` | Summary + risk metrics |

<br>

## Features

### AI-Powered News Sentiment Analysis
- **GPT-4o-mini sentiment engine** analyzes Google News headlines with financial context understanding
- Understands nuance: "profit booking" = NEGATIVE, "earnings beat" = POSITIVE
- Handles negation: "not expected to rise" = NEGATIVE
- Returns sentiment (POSITIVE/NEGATIVE/NEUTRAL), confidence score (0-1), trading impact, and reasoning
- **Automatic fallback** to keyword matching if OpenAI API is unavailable
- Integrated into Scanner and Analyst -- stocks with NEGATIVE sentiment (confidence > 0.6) are auto-rejected

### Bull/Bear Debate Workflow
- Inspired by academic research on adversarial agent architectures
- **Bull Agent** presents the strongest case for buying with real technical data
- **Bear Agent** presents risks, overbought conditions, and what could go wrong
- **Judge Agent** weighs both arguments and makes a GO/NO-GO decision
- If Judge says NO-GO, the trade is killed before reaching the Risk Manager
- Reduces false positives and prevents emotional/momentum-driven entries

### Live Trading
- **Groww API Integration** -- Places real MARKET orders on NSE via the Groww broker
- **MIS (Margin Intraday)** product type for intraday with up to 5x margin
- **Order tracking** with fill price verification and order status polling
- **Both BUY and SELL** execution with full audit trail
- **Margin-aware position sizing** -- knows actual capital vs effective buying power

### Technical Analysis Engine
10+ indicators computed on 5-minute candles using `pandas-ta`:

| Indicator | Signal Logic |
|-----------|-------------|
| **SMA** (20/50/200) | Price above = BUY, below = SELL |
| **EMA Crossover** (9/21) | Golden cross = BUY, Death cross = SELL |
| **RSI** (14) | <30 Oversold BUY, >70 Overbought SELL |
| **MACD** | Bullish crossover = BUY, Histogram direction |
| **Bollinger Bands** | Lower band = BUY, Upper band = SELL, %B zones |
| **Stochastic** (K/D) | <20 Oversold BUY, >80 Overbought SELL |
| **VWAP** | Above = Bullish intraday, Below = Bearish |
| **ADX** (14) | >25 Strong trend, DI+/DI- for direction |
| **ATR** (14) | Volatility assessment for position sizing |
| **Volume** | Volume confirmation/divergence detection |

**Candlestick Pattern Detection:**
Doji, Hammer, Inverted Hammer, Shooting Star, Marubozu, Bullish/Bearish Engulfing, Morning Star, Evening Star

**Volume Confirmation:**
- Current volume vs 20-period average -- rejects low-volume rallies
- Detects volume divergence (price up + low volume = weak move, likely to reverse)
- 20% strength penalty for BUY signals without volume confirmation

**Signal Scoring:**
- Requires 3+ BUY indicators (minimum), 60% majority, trend alignment
- Capped at 0.85 strength to prevent overconfidence
- No artificial multipliers -- uses raw indicator agreement ratio

### 5-Layer Trade Validation
Every trade passes through 5 independent gates. Note which ones are enforced in
code rather than in a prompt -- after the incident described in
[Results](#results-what-actually-happened), every limit that can lose money is.

```
Layer 1: Scanner            -- Top gainers only (up 1-5% today), strength >= 0.65, volume confirmed
Layer 2: Analyst/save_signal-- CODE: engine re-verifies the signal and overrides the LLM's claimed strength
Layer 3: Bull/Bear Debate   -- Judge must return GO before the risk manager sees the trade
Layer 4: Risk Manager       -- CODE: _enforce_risk_limits() clamps quantity; daily-loss circuit breaker
Layer 5: place_trade Tool   -- CODE: symbol cooldown, SL required, target/SL bounds, live signal re-check
```

### Position Management
- **Trailing Stop-Loss** -- Tracks the peak price after entry, auto-sells on reversal
- **Target Exit** -- Auto-sells when target price is reached
- **Position Monitor** -- Checks every 2 minutes during market hours
- **Auto Square-Off** -- Sells ALL intraday positions at 3:15 PM (before 3:30 market close)

### Risk Management

All of the following are enforced in code, not in an LLM prompt:

- **Risk-per-trade sizing (primary control)** -- quantity is sized so that a
  stop-out costs at most `RISK_PER_TRADE_PCT` of *real* capital (default 0.5%)
- Notional cap per position as a % of **real** capital, not margin buying power
  (default 20%) -- this is the limit whose earlier misreading allowed Rs 12.5L
  into a single name
- Maximum daily loss limit (default 2%), as a circuit breaker in both
  `place_trade` and the scheduler cycle
- Daily profit target (default 1.5% -- stops taking new trades when hit)
- Maximum open positions (default 10)
- 3-day per-symbol cooldown after a loss on that symbol
- Stale positions carried past their trading day are squared off
- Risk-reward ratio enforcement and a news sentiment gate (prompt-level)

### Risk-Adjusted Performance Metrics
- **Sharpe Ratio** -- Annualized risk-adjusted returns (India risk-free rate ~7%)
- **Win Rate** -- Percentage of profitable trades
- **Profit Factor** -- Gross profit / Gross loss
- **Average Win / Average Loss** -- Per-trade performance
- **Max Drawdown** -- Largest peak-to-trough decline with visual chart

### Dashboard (Streamlit UI)

| Tab | What It Shows |
|-----|--------------|
| **Portfolio** | Live portfolio value, P&L, positions, trailing SL status, Sharpe ratio, drawdown chart |
| **Trades** | Complete trade history with timestamps, prices, rationale (filtered by paper/live mode) |
| **Signals** | AI-generated signals with strength and reasoning |
| **Analysis** | Interactive stock analyzer with candlestick charts, support/resistance |
| **Top Movers** | Multi-index scanner (NIFTY 50/100/500/Midcap 100/Smallcap 100/Total Market) with "Analyze Selected" |
| **Agent Logs** | Color-coded activity logs from all agents |
| **Live Activity** | Real-time pipeline and position monitor status |
| **Watchlist** | Custom stock watchlist with live prices |

**Sidebar Controls:** Groww API keys, OpenAI settings, trading mode (Paper/Live), scheduler start/stop, manual pipeline trigger, square-off button, risk parameters, margin multiplier.

<br>

## Project Structure

```
alpha-prime/
|
|-- streamlit_app.py              # Hosted demo entrypoint (forces DEMO_MODE on)
|-- run_dashboard.py              # Launch Streamlit UI (live system)
|-- run_trading.py                # Launch headless trading scheduler
|-- run_analysis.py               # Quick AI stock analysis CLI
|-- run_backtest.py               # Backtest CLI (--symbols/--universe/--period/--compare)
|-- run_sweep.py                  # Parameter sweep over SL/target/trailing stop
|-- requirements.txt              # Dashboard/demo dependencies (what the host installs)
|-- requirements-full.txt         # Full live system dependencies
|-- .env.example                  # Documented config template (.env is gitignored)
|
|-- alpha_prime/
|   |-- __init__.py
|   |
|   |-- agents/                   # Multi-Agent AI Pipeline
|   |   |-- pipeline.py           # LangGraph state machine (7 agents + debate)
|   |   |-- prompts.py            # System prompts (scanner, analyst, bull, bear, judge, risk, exec, portfolio)
|   |   |-- tools.py              # 16 LangChain tools + _enforce_risk_limits() code-side clamps
|   |
|   |-- core/                     # System Core
|   |   |-- config.py             # Pydantic settings, DEMO_MODE, credential-write guards
|   |   |-- risk.py               # Pure sizing/trailing-stop functions, shared by live AND backtest
|   |   |-- broker.py             # Groww API broker + Paper trading broker, per-SELL P&L
|   |   |-- database.py           # SQLite database (trades, signals, logs)
|   |   |-- market_hours.py       # NSE session-time helpers (no market-data dependency)
|   |   |-- scheduler.py          # APScheduler + Position Monitor + Auto Square-Off
|   |
|   |-- data/                     # Market Data & Analysis
|   |   |-- market_data.py        # yfinance integration, multi-index stock lists
|   |   |-- technical_analysis.py # 10+ indicators, volume confirmation, candlestick patterns
|   |
|   |-- backtest/                 # Backtesting
|   |   |-- backtester.py         # Event-driven, no-lookahead, Indian intraday cost model
|   |
|   |-- ui/                       # User Interface
|       |-- dashboard.py          # Streamlit dashboard (9 tabs, risk metrics, drawdown chart)
|
|-- scripts/
|   |-- build_demo_db.py          # Builds the public snapshot from the live DB
|
|-- tests/
|   |-- test_live_pnl.py          # P&L attribution tests (stateful mock broker)
|
|-- backtest_results/             # Committed backtest output (trades, equity, sweep)
|
|-- data/
|   |-- alpha_prime.db            # Live SQLite database (gitignored)
|   |-- demo/
|       |-- alpha_prime_demo.db   # Committed read-only snapshot for the demo
|
|-- logs/
    |-- trading_*.log             # Rotating daily logs
```

### Risk logic is shared, not duplicated

`core/risk.py` holds `compute_max_quantity` and `compute_trailed_stop` as pure
functions. The live execution path and the backtester both call them, so a
backtest cannot silently diverge from what the live system would actually do --
a common way for backtests to flatter themselves.

<br>

## Getting Started

### Prerequisites

- **Python 3.13+**
- **Groww Trading Account** with API access ([trade-api](https://groww.in/trade-api))
- **OpenAI API Key** ([platform.openai.com](https://platform.openai.com/))

### Installation

```bash
# Clone the repository
git clone https://github.com/Sreenivasan2002/Alpha-Prime.git
cd Alpha-Prime

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies for the full live system
pip install -r requirements-full.txt
```

There are two dependency sets:

| File | Contents | Use |
|---|---|---|
| `requirements.txt` | streamlit, pandas, numpy, plotly, pydantic (9 packages) | The read-only dashboard. This is what the hosted demo installs -- Streamlit Cloud auto-discovers `requirements.txt`. |
| `requirements-full.txt` | the above plus langchain/langgraph, openai, growwapi, yfinance, pandas-ta, apscheduler | The live trading system. |

The split exists because the demo never calls an LLM, a broker or a market-data
provider, so shipping those to the host would add install time and memory
pressure for nothing. In demo mode the dashboard does not import them at all.

To browse the demo locally without any credentials:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Configuration

Create a `.env` file in the project root:

```env
# Groww API (get from https://groww.in/trade-api)
GROWW_API_KEY=your_groww_jwt_token
GROWW_SECRET_KEY=your_groww_secret

# OpenAI
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.1

# Trading Parameters
TRADING_MODE=paper              # paper or live
TRADING_CAPITAL=10000           # Actual capital in Rs
MARGIN_MULTIPLIER=5.0           # Broker margin (e.g., 5x on Groww)
MAX_DAILY_LOSS_PCT=2.0          # Max loss per day (%)
DAILY_PROFIT_TARGET_PCT=1.5     # Stop trading after this profit (%)
MAX_POSITION_SIZE_PCT=25.0      # Max capital per trade (% of effective capital)
MAX_OPEN_POSITIONS=3            # Max simultaneous trades
DEFAULT_STOP_LOSS_PCT=0.8       # Default stop-loss (%)
DEFAULT_TARGET_PCT=1.2          # Default target (%)
MIN_SIGNAL_STRENGTH=0.7         # Minimum signal strength to trade

# Scheduler
ANALYSIS_INTERVAL_MINUTES=15    # How often the pipeline runs
```

> **Important:** Groww API keys expire daily. Update `GROWW_API_KEY` each morning before market open, or use the dashboard sidebar to update keys at runtime.

### Running

#### Option 1: Dashboard UI (Recommended)

```bash
venv\Scripts\streamlit run run_dashboard.py
```

Opens a web dashboard at `http://localhost:8501`. Click **Start** in the Scheduler section to begin automated trading.

#### Option 2: Headless Mode (No UI)

```bash
venv\Scripts\python run_trading.py
```

Runs the trading scheduler in the terminal. Press `Ctrl+C` to stop.

#### Option 3: Quick Analysis

```bash
# Analyze specific stocks
venv\Scripts\python run_analysis.py RELIANCE TCS INFY

# Scan market for opportunities
venv\Scripts\python run_analysis.py
```

<br>

## Daily Trading Schedule

```
  TIME (IST)    EVENT                        DESCRIPTION
  ----------    -----                        -----------
   9:00 AM      Pre-Market Scan              AI scans market before open
   9:15 AM      Market Opens                 First trading cycle begins
   9:15-3:00    Trading Cycles               Pipeline runs every 15 min:
                                              - Scans stocks + news sentiment
                                              - Deep technical analysis
                                              - Bull/Bear debate on picks
                                              - Risk-managed position sizing
                                              - Executes BUY orders
   9:15-3:15    Position Monitor             Checks every 2 min:
                                              - Trailing stop-loss
                                              - Target hit detection
                                              - Auto-sells on triggers
   3:15 PM      Auto Square-Off              Sells ALL open positions
   3:25 PM      End-of-Day Review            AI reviews day's performance
   3:30 PM      Market Closes                System goes idle
```

<br>

## How Trades Are Executed

Here is the exact flow when Alpha-Prime places a trade:

```
1. Scanner identifies KOTAKBANK as BUY candidate
   - Technical: overall_signal=BUY, strength=0.72, trend=BULLISH
   - News: check_news_sentiment -> POSITIVE (75% confidence)
                    |
2. Analyst confirms with deep analysis:
   - 314 candles (5-min, 5 days)
   - 10+ indicators computed (including volume confirmation)
   - 6 BUY vs 2 SELL = BUY signal, volume_confirmed=True
   - News sentiment: POSITIVE, no negative headlines
   - Entry: Rs 430, SL: Rs 426.56, Target: Rs 435.16
                    |
3. Bull/Bear Debate:
   - Bull Agent: "Strong momentum, above VWAP, volume 1.4x average, positive news"
   - Bear Agent: "Resistance at 435, RSI approaching 60, limited upside"
   - Judge: GO (Bull Score: 7, Bear Score: 4)
                    |
4. Risk Manager checks:
   - Actual Capital: Rs 10,000 (5x margin = Rs 50,000 buying power)
   - Max per position: 25% = Rs 12,500
   - Quantity: floor(12500 / 430) = 29 shares
   - Open positions: 0 < 3 max
   - R:R ratio: 1:1.5 (OK)
   - APPROVED: BUY KOTAKBANK qty=29
                    |
5. place_trade hard-coded validation (Layer 4):
   - SL defined? Yes. Target >= 0.5%? Yes. SL <= 3%? Yes.
   - Re-runs live analysis: signal still BUY, not BEARISH
   - PASSED
                    |
6. Executor calls Groww API:
   - place_order(trading_symbol="KOTAKBANK", product="MIS", ...)
   - Response: { groww_order_id: "GMK...", order_status: "EXECUTED" }
                    |
7. Position Monitor registers:
   - Entry: Rs 430, Peak: Rs 430
   - Trailing SL: Rs 426.56
   - Target: Rs 435.16
                    |
8. Every 2 minutes, monitor checks live price:
   - Price rises to Rs 433 -> Peak updates, SL trails up
   - Price drops below SL -> AUTO SELL triggered
   - Price hits target -> TARGET HIT, AUTO SELL
                    |
9. At 3:15 PM: Square-off sells any remaining positions
```

<br>

## The 15 AI Tools

Each agent has access to a curated set of tools:

### Market Analysis Tools
| Tool | Description |
|------|-------------|
| `get_stock_price` | Get real-time price for any NSE stock |
| `get_multiple_stock_prices` | Batch price fetch (comma-separated symbols) |
| `get_stock_info` | Detailed stock info (market cap, sector, P/E) |
| `get_historical_data` | OHLCV data with configurable period/interval |
| `run_technical_analysis` | Full TA on daily timeframe (10+ indicators) |
| `run_intraday_analysis` | Full TA on 5-min candles (5 days of data) |
| `get_nifty50_stocks` | List of NIFTY 50 constituent symbols |
| `check_market_status` | Is market open? Pre-market? Time to open? |
| `check_news_sentiment` | **AI-powered** news sentiment via GPT-4o-mini on Google News headlines |

### Trading Tools
| Tool | Description |
|------|-------------|
| `place_trade` | Place BUY/SELL order via Groww API (with hard-coded 4-layer validation) |
| `get_portfolio` | Portfolio summary: cash, invested, P&L, positions |
| `get_current_positions` | All open intraday positions with live P&L |
| `get_recent_trades` | Trade history from database |

### Signal Tools
| Tool | Description |
|------|-------------|
| `save_signal` | Record an AI trading signal to database |
| `get_recent_signals` | Fetch recent signals for review |

<br>

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **AI/LLM** | OpenAI GPT-4o-mini (reasoning + sentiment analysis) |
| **Agent Orchestration** | LangGraph (state machine with 7 agents), LangChain |
| **Debate Workflow** | Bull/Bear adversarial agents + Judge (inspired by academic research) |
| **News Sentiment** | GPT-4o-mini on Google News RSS (with keyword fallback) |
| **Broker** | Groww API (growwapi SDK) with margin support |
| **Market Data** | yfinance (multi-index: NIFTY 50/100/500/Midcap/Smallcap) |
| **Technical Analysis** | pandas-ta (10+ indicators + volume confirmation) |
| **UI** | Streamlit + Plotly charts (8 tabs, drawdown visualization) |
| **Database** | SQLite with WAL mode (7 tables, paper/live separation) |
| **Scheduling** | APScheduler (cron + interval triggers) |
| **Config** | Pydantic Settings + python-dotenv |
| **Logging** | Loguru (daily rotation, 30-day retention) |

<br>

## Database Schema

Alpha-Prime uses SQLite with 7 tables:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `trades` | Complete trade history | symbol, action, quantity, price, status, order_id, rationale, trading_mode |
| `signals` | AI-generated signals | symbol, signal_type, strength, reasoning |
| `agent_logs` | Agent activity logs | agent_name, log_type, message, details |
| `portfolio` | Portfolio snapshots | capital, invested, total_value, daily_pnl |
| `watchlist` | User watchlist | symbol, sector, notes |
| `market_cache` | Data cache (5-min TTL) | cache_key, data_json |
| `system_state` | Key-value state store | key, value (day_start_portfolio_value, scheduler status, etc.) |

<br>

## Submitted To

**AWS AI for Bharat Hackathon 2025** -- Autonomous AI Trading Agent category.

<br>

## Disclaimer

> **This software is for educational and research purposes only.** Trading in financial markets involves substantial risk of loss. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through the use of this system. Always understand the risks before trading with real money.

<br>

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with LangGraph + OpenAI GPT-4o-mini + Groww API | Multi-Agent Debate Architecture</sub>
</div>
