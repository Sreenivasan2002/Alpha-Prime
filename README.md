<div align="center">

# ALPHA-PRIME

### Autonomous Intraday Trading System for Indian Markets

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-blueviolet.svg)](https://langchain-ai.github.io/langgraph/)
[![OpenAI GPT-4o](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com/)
[![Groww API](https://img.shields.io/badge/Broker-Groww%20API-orange.svg)](https://groww.in/trade-api)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)

**A fully autonomous AI-powered intraday trading system that scans the NSE market, analyzes stocks using 9+ technical indicators, makes risk-managed decisions, and executes live trades through the Groww broker API -- all without human intervention.**

[Getting Started](#getting-started) |
[Architecture](#architecture) |
[Pipeline](#the-multi-agent-pipeline) |
[Features](#features) |
[Configuration](#configuration)

---

</div>

<br>

## How It Works

Alpha-Prime uses a **5-agent AI pipeline** built on LangGraph where each agent is a specialized GPT-4o instance with access to real-time market tools. Every 15 minutes during market hours, the pipeline runs automatically:

```
Market Opens (9:15 AM)
       |
       v
  +---------+     +-----------+     +--------------+     +-----------+     +------------------+
  | SCANNER |---->| ANALYST   |---->| RISK MANAGER |---->| EXECUTOR  |---->| PORTFOLIO REVIEW |
  |         |     |           |     |              |     |           |     |                  |
  | Scans   |     | Deep TA   |     | Position     |     | Places    |     | Tracks P&L,      |
  | NIFTY50 |     | on picks  |     | sizing &     |     | live      |     | reviews trades,  |
  | stocks  |     | Entry/SL/ |     | approval     |     | orders    |     | recommendations  |
  |         |     | Target    |     |              |     | on Groww  |     |                  |
  +---------+     +-----------+     +--------------+     +-----------+     +------------------+
       |                                                       |
       | Every 15 min                                          | Every 2 min
       | during market hours                                   | Position Monitor
       |                                                       | checks SL/Target
       v                                                       v
  Auto Square-Off at 3:15 PM                          Trailing Stop-Loss
  (sells all positions)                               (auto-sells on reversal)
```

<br>

## The Multi-Agent Pipeline

The core of Alpha-Prime is a **LangGraph state machine** where 5 AI agents work in sequence. Each agent has specialized tools and a focused mission.

### Pipeline Architecture

```
                          +------------------------------------------+
                          |          TradingState (shared)           |
                          |                                          |
                          |  scanner_output   | analyst_output       |
                          |  risk_output      | execution_output     |
                          |  portfolio_output | current_phase        |
                          |  trade_decisions  | error                |
                          +------------------------------------------+
                                           |
                     +---------------------+---------------------+
                     |                     |                     |
                     v                     v                     v
              LangGraph StateGraph    OpenAI GPT-4o       14 LangChain Tools
              (orchestration)         (reasoning)         (market actions)

  ==========================================================================

  PHASE 1                    PHASE 2                    PHASE 3
  MARKET SCANNER             TECHNICAL ANALYST          RISK MANAGER
  +-----------------+        +-----------------+        +-----------------+
  |                 |        |                 |        |                 |
  | Tools:          |        | Tools:          |        | Tools:          |
  | - check_market  |  --->  | - run_intraday  |  --->  | - get_portfolio |
  | - get_prices    |        |   _analysis     |        | - get_positions |
  | - run_intraday  |        | - get_price     |        | - get_prices    |
  |   _analysis     |        | - save_signal   |        |                 |
  |                 |        |                 |        | Calculates:     |
  | Scans top 10    |        | For each pick:  |        | - Position size |
  | NIFTY 50 stocks |        | - Entry price   |        | - Risk/reward   |
  | Identifies 3    |        | - Stop-loss     |        | - Max exposure  |
  | BUY candidates  |        | - Target price  |        |                 |
  |                 |        | - Strength      |        | APPROVE/REJECT  |
  +-----------------+        +-----------------+        +-----------------+
                                                               |
                                                               v
  PHASE 4                                              PHASE 5
  EXECUTION AGENT                                      PORTFOLIO MANAGER
  +------------------+                                  +------------------+
  |                  |                                  |                  |
  | Tools:           |                                  | Tools:           |
  | - place_trade    |  ----> Live Order on Groww --->  | - get_portfolio  |
  | - get_price      |        (MIS/Intraday)            | - get_positions  |
  |                  |                                  | - get_trades     |
  | For each         |                                  |                  |
  | APPROVED trade:  |                                  | Reviews:         |
  | - Verify price   |                                  | - All positions  |
  | - Execute BUY    |                                  | - Day's P&L      |
  | - Confirm fill   |                                  | - Recommendations|
  |                  |                                  |                  |
  +------------------+                                  +------------------+
```

### Agent Details

| # | Agent | Role | Tools | Output |
|---|-------|------|-------|--------|
| 1 | **Market Scanner** | Scans NIFTY 50 for momentum setups | `check_market_status`, `get_multiple_stock_prices`, `run_intraday_analysis` | Top 3 BUY candidates |
| 2 | **Technical Analyst** | Deep analysis with 9+ indicators | `run_intraday_analysis`, `get_stock_price`, `save_signal` | Entry, Stop-Loss, Target for each stock |
| 3 | **Risk Manager** | Position sizing and trade approval | `get_portfolio`, `get_current_positions`, `get_stock_price` | APPROVED trades with quantity |
| 4 | **Execution Agent** | Places live orders on Groww | `place_trade`, `get_stock_price` | Order confirmations |
| 5 | **Portfolio Manager** | Reviews session and tracks P&L | `get_portfolio`, `get_current_positions`, `get_recent_trades` | Summary and recommendations |

<br>

## Features

### Live Trading
- **Groww API Integration** -- Places real MARKET orders on NSE via the Groww broker
- **MIS (Margin Intraday)** product type for intraday square-off
- **Order tracking** with fill price verification and order status polling
- **Both BUY and SELL** execution with full audit trail

### Technical Analysis Engine
9+ indicators computed on 5-minute candles using `pandas-ta`:

| Indicator | Signal Logic |
|-----------|-------------|
| **SMA** (20/50/200) | Price above = BUY, below = SELL |
| **EMA Crossover** (9/21) | Golden cross = BUY, Death cross = SELL |
| **RSI** (14) | <30 Oversold BUY, >70 Overbought SELL, 30-45 Mild BUY, 55-70 Mild SELL |
| **MACD** | Bullish crossover = BUY, Histogram direction |
| **Bollinger Bands** | Lower band = BUY, Upper band = SELL, %B zones |
| **Stochastic** (K/D) | <20 Oversold BUY, >80 Overbought SELL, extended zones |
| **VWAP** | Above = Bullish intraday, Below = Bearish |
| **ADX** (14) | >25 Strong trend, DI+/DI- for direction |
| **ATR** (14) | Volatility assessment for position sizing |

**+ Candlestick Pattern Detection:**
Doji, Hammer, Inverted Hammer, Shooting Star, Marubozu, Bullish/Bearish Engulfing, Morning Star, Evening Star

**Signal Voting:** Count-based majority -- if more indicators say BUY than SELL, the stock is a BUY.

### Position Management
- **Trailing Stop-Loss** -- Tracks the peak price after entry, auto-sells if price drops 1.5% from peak
- **Target Exit** -- Auto-sells when target price (default +3%) is reached
- **Position Monitor** -- Checks every 2 minutes during market hours
- **Auto Square-Off** -- Sells ALL intraday positions at 3:15 PM (before 3:30 market close)

### Risk Management
- Maximum daily loss limit (configurable, default 3%)
- Maximum position size as % of capital (configurable, default 10%)
- Maximum number of open positions (configurable, default 3)
- Default stop-loss: 1.5% from entry
- Default target: 3.0% from entry
- Risk-reward ratio enforcement

### Dashboard (Streamlit UI)

| Tab | What It Shows |
|-----|--------------|
| **Portfolio** | Live portfolio value, P&L, available cash, positions with trailing SL status |
| **Trades** | Complete trade history with timestamps, prices, rationale |
| **Signals** | AI-generated signals with strength and reasoning |
| **Analysis** | Interactive stock analyzer with candlestick charts, support/resistance |
| **Agent Logs** | Color-coded activity logs from all 5 agents |
| **Watchlist** | Custom stock watchlist with live prices |

**Sidebar Controls:** Groww API keys, OpenAI settings, trading mode (Paper/Live), scheduler start/stop, manual pipeline trigger, square-off button, risk parameters.

<br>

## Project Structure

```
alpha-prime/
|
|-- run_dashboard.py              # Launch Streamlit UI
|-- run_trading.py                # Launch headless trading scheduler
|-- run_analysis.py               # Quick AI stock analysis CLI
|-- requirements.txt              # Python dependencies
|-- .env                          # API keys & trading config
|
|-- alpha_prime/
|   |-- __init__.py               # v1.0.0
|   |
|   |-- agents/                   # Multi-Agent AI Pipeline
|   |   |-- pipeline.py           # LangGraph state machine (5 agents)
|   |   |-- prompts.py            # System prompts for each agent
|   |   |-- tools.py              # 14 LangChain tools for market actions
|   |
|   |-- core/                     # System Core
|   |   |-- config.py             # Pydantic settings (.env parsing)
|   |   |-- broker.py             # Groww API broker (order placement)
|   |   |-- database.py           # SQLite database (trades, signals, logs)
|   |   |-- scheduler.py          # APScheduler + Position Monitor
|   |
|   |-- data/                     # Market Data & Analysis
|   |   |-- market_data.py        # yfinance integration, market hours
|   |   |-- technical_analysis.py # 9+ indicators, candlestick patterns
|   |
|   |-- ui/                       # User Interface
|       |-- dashboard.py          # Streamlit dashboard (6 tabs)
|
|-- data/
|   |-- alpha_prime.db            # SQLite database file
|
|-- logs/
    |-- trading_*.log             # Rotating daily logs
```

<br>

## Getting Started

### Prerequisites

- **Python 3.13+**
- **Groww Trading Account** with API access ([trade-api](https://groww.in/trade-api))
- **OpenAI API Key** ([platform.openai.com](https://platform.openai.com/))

### Installation

```bash
# Clone or navigate to the project
cd "Algo Trading"

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
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
TRADING_CAPITAL=5000            # Total capital in Rs
MAX_DAILY_LOSS_PCT=3.0          # Max loss per day (%)
MAX_POSITION_SIZE_PCT=40.0      # Max capital per trade (%)
MAX_OPEN_POSITIONS=3            # Max simultaneous trades
DEFAULT_STOP_LOSS_PCT=1.5       # Default stop-loss (%)
DEFAULT_TARGET_PCT=3.0          # Default target (%)

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
   9:15-3:00    Trading Cycles               Pipeline runs every 15 min
                                              - Scans stocks
                                              - Analyzes technicals
                                              - Sizes positions
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
                    |
2. Analyst runs intraday analysis:
   - 314 candles (5-min, 5 days)
   - 9 indicators computed
   - 5 BUY vs 3 SELL = BUY signal
   - Entry: Rs 430, SL: Rs 423.55, Target: Rs 438.60
                    |
3. Risk Manager checks:
   - Capital: Rs 5,000
   - Max per position: 40% = Rs 2,000
   - Quantity: floor(2000 / 430) = 4 shares
   - Open positions: 0 < 3 max
   - APPROVED: BUY KOTAKBANK qty=4
                    |
4. Executor calls Groww API:
   - place_order(
       trading_symbol="KOTAKBANK",
       transaction_type="BUY",
       product="MIS",           <-- Intraday
       order_type="MARKET",
       quantity=4,
       exchange="NSE",
       segment="CASH",
       validity="DAY"
     )
   - Response: { groww_order_id: "GMK...", order_status: "EXECUTED" }
                    |
5. Position Monitor registers:
   - Entry: Rs 430, Peak: Rs 430
   - Trailing SL: Rs 423.55 (1.5% below)
   - Target: Rs 438.60 (2% above)
                    |
6. Every 2 minutes, monitor checks live price:
   - If price rises to Rs 435 -> Peak updates, SL moves to Rs 428.48
   - If price drops below SL -> AUTO SELL triggered
   - If price hits Rs 438.60 -> TARGET HIT, AUTO SELL
                    |
7. At 3:15 PM: Square-off sells any remaining positions
```

<br>

## The 14 AI Tools

Each agent has access to a curated set of tools:

### Market Analysis Tools
| Tool | Description |
|------|-------------|
| `get_stock_price` | Get real-time price for any NSE stock |
| `get_multiple_stock_prices` | Batch price fetch (comma-separated symbols) |
| `get_stock_info` | Detailed stock info (market cap, sector, P/E) |
| `get_historical_data` | OHLCV data with configurable period/interval |
| `run_technical_analysis` | Full TA on daily timeframe (9+ indicators) |
| `run_intraday_analysis` | Full TA on 5-min candles (5 days of data) |
| `get_nifty50_stocks` | List of NIFTY 50 constituent symbols |
| `check_market_status` | Is market open? Pre-market? Time to open? |

### Trading Tools
| Tool | Description |
|------|-------------|
| `place_trade` | Place BUY/SELL order via Groww API (auto-registers with position monitor) |
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
| **AI/LLM** | OpenAI GPT-4o / GPT-4o-mini |
| **Agent Orchestration** | LangGraph (state machine), LangChain |
| **Broker** | Groww API (growwapi SDK) |
| **Market Data** | yfinance |
| **Technical Analysis** | pandas-ta (9+ indicators) |
| **UI** | Streamlit + Plotly charts |
| **Database** | SQLite (7 tables) |
| **Scheduling** | APScheduler (cron + interval triggers) |
| **Config** | Pydantic Settings + python-dotenv |
| **Logging** | Loguru (daily rotation, 30-day retention) |

<br>

## Database Schema

Alpha-Prime uses SQLite with 7 tables:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `trades` | Complete trade history | symbol, action, quantity, price, status, order_id, rationale |
| `signals` | AI-generated signals | symbol, signal_type, strength, reasoning |
| `agent_logs` | Agent activity logs | agent_name, log_type, message, details |
| `portfolio` | Portfolio snapshots | capital, invested, total_value, daily_pnl |
| `watchlist` | User watchlist | symbol, sector, notes |
| `market_cache` | Data cache (5-min TTL) | cache_key, data_json |
| `system_state` | Key-value state store | key, value (scheduler status, etc.) |

<br>

## Disclaimer

> **This software is for educational and research purposes only.** Trading in financial markets involves substantial risk of loss. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through the use of this system. Always understand the risks before trading with real money.

<br>

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with LangGraph + OpenAI + Groww API</sub>
</div>
