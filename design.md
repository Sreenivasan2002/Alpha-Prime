<div align="center">

# System Design Document

### Alpha-Prime | AI-Powered Autonomous Intraday Trading System

<br>

**AWS AI for Bharat Hackathon 2025**

---

`Version 1.0` &nbsp;&nbsp; | &nbsp;&nbsp; `Last Updated: February 2025` &nbsp;&nbsp; | &nbsp;&nbsp; `Team: Alpha-Prime`

---

</div>

<br>

## Table of Contents

| # | Section | Description |
|---|---------|-------------|
| 1 | [System Architecture](#1-system-architecture) | High-level architecture and component overview |
| 2 | [Multi-Agent Pipeline Design](#2-multi-agent-pipeline-design) | LangGraph state machine and agent specifications |
| 3 | [Technical Analysis Engine](#3-technical-analysis-engine) | Indicator computation and signal generation |
| 4 | [Broker Integration Layer](#4-broker-integration-layer) | Groww API integration and order management |
| 5 | [Position Management System](#5-position-management-system) | Trailing stop-loss, targets, and auto square-off |
| 6 | [Database Design](#6-database-design) | Schema, tables, and data flow |
| 7 | [Scheduling & Automation](#7-scheduling--automation) | APScheduler jobs and daily timeline |
| 8 | [Dashboard UI Design](#8-dashboard-ui-design) | Streamlit interface and components |
| 9 | [Tool System Design](#9-tool-system-design) | LangChain tools available to AI agents |
| 10 | [Configuration & Security](#10-configuration--security) | Environment setup and credential management |
| 11 | [Error Handling & Resilience](#11-error-handling--resilience) | Failure modes and recovery strategies |
| 12 | [AWS Deployment Architecture](#12-aws-deployment-architecture) | Cloud deployment and scaling plan |

---

<br>

## 1. System Architecture

### 1.1 High-Level Architecture

The system follows a **5-layer architecture** with clear separation of concerns:

```
+=====================================================================+
|                       USER INTERFACE LAYER                           |
|  +---------------------------------------------------------------+  |
|  |              Streamlit Dashboard (dashboard.py)                |  |
|  |   Portfolio  |  Trades  |  Signals  |  Analysis  |  Logs      |  |
|  +---------------------------------------------------------------+  |
+=====================================================================+
                               |
                        HTTP / WebSocket
                               |
+=====================================================================+
|                       APPLICATION LAYER                              |
|  +---------------------------------------------------------------+  |
|  |               Scheduler (scheduler.py)                         |  |
|  |   Trading Pipeline (5-15 min)  |  Position Monitor (2 min)    |  |
|  |   Pre-Market Scan (9:00 AM)    |  Auto Square-Off (3:15 PM)   |  |
|  +---------------------------------------------------------------+  |
|                               |                                      |
|  +---------------------------------------------------------------+  |
|  |          Multi-Agent Pipeline (pipeline.py)                    |  |
|  |                                                                |  |
|  |   [Scanner] --> [Analyst] --> [Risk Mgr] --> [Executor]        |  |
|  |                                                                |  |
|  |              LangGraph State Machine + GPT-4o                  |  |
|  +---------------------------------------------------------------+  |
+=====================================================================+
                               |
                               |
+=====================================================================+
|                        SERVICE LAYER                                 |
|  +-----------------+  +------------------+  +--------------------+  |
|  |    Broker API   |  |   Market Data    |  | Technical Analysis |  |
|  |  (broker.py)    |  | (market_data.py) |  | (technical_        |  |
|  |                 |  |                  |  |    analysis.py)    |  |
|  |  Groww API      |  |  yfinance API    |  |  pandas-ta +       |  |
|  |  MIS Orders     |  |  OHLCV Candles   |  |  scipy + numpy     |  |
|  +-----------------+  +------------------+  +--------------------+  |
+=====================================================================+
                               |
                               |
+=====================================================================+
|                         DATA LAYER                                   |
|  +-----------------+  +------------------+  +--------------------+  |
|  |    SQLite DB    |  |   Config Mgmt    |  |    Logging         |  |
|  | (database.py)   |  |  (config.py)     |  |   (loguru)         |  |
|  |                 |  |                  |  |                    |  |
|  |  7 Tables       |  |  Pydantic +      |  |  Rotating daily    |  |
|  |  SQLAlchemy ORM |  |  .env file       |  |  30-day retention  |  |
|  +-----------------+  +------------------+  +--------------------+  |
+=====================================================================+
                               |
                               |
+=====================================================================+
|                     EXTERNAL SERVICES                                |
|  +-----------------+  +------------------+  +--------------------+  |
|  |   OpenAI API    |  |    Groww API     |  |    yfinance        |  |
|  |   (GPT-4o)      |  |   (Brokerage)    |  |  (Market Data)     |  |
|  |                 |  |                  |  |                    |  |
|  |  Agent          |  |  Order           |  |  Real-time +       |  |
|  |  Reasoning      |  |  Execution       |  |  Historical        |  |
|  +-----------------+  +------------------+  +--------------------+  |
+=====================================================================+
```

### 1.2 Component Overview

| Layer | Components | Responsibility | Key Files |
|-------|-----------|----------------|-----------|
| **UI** | Streamlit Dashboard | User interaction, visualization, controls | `dashboard.py` |
| **Application** | Scheduler, Pipeline | Orchestration, automation, agent coordination | `scheduler.py`, `pipeline.py` |
| **Service** | Broker, Market Data, TA | Business logic, external integrations | `broker.py`, `market_data.py`, `technical_analysis.py` |
| **Data** | Database, Config, Logs | Persistence, configuration, audit trail | `database.py`, `config.py` |
| **External** | OpenAI, Groww, yfinance | AI reasoning, order execution, market data | Third-party APIs |

### 1.3 Project Structure

```
alpha_prime/
|
+-- agents/
|   +-- pipeline.py          # LangGraph multi-agent state machine
|   +-- prompts.py           # Agent system prompts (mission-driven)
|   +-- tools.py             # 14 LangChain tools for agents
|
+-- core/
|   +-- broker.py            # Groww API integration (MIS orders)
|   +-- scheduler.py         # APScheduler + PositionMonitor class
|   +-- database.py          # SQLAlchemy ORM (7 tables)
|   +-- config.py            # Pydantic settings from .env
|
+-- data/
|   +-- market_data.py       # yfinance wrapper with caching
|   +-- technical_analysis.py # 9+ indicators + pattern recognition
|
+-- ui/
|   +-- dashboard.py         # Streamlit web dashboard (6 tabs)
|
+-- .env                     # Configuration (API keys, risk params)
+-- main.py                  # Application entry point
+-- requirements.txt         # Python dependencies
```

---

<br>

## 2. Multi-Agent Pipeline Design

### 2.1 LangGraph State Machine

The pipeline uses **LangGraph's StateGraph** to orchestrate 4 specialized AI agents in a linear flow. Each agent is an LLM-powered node with access to specific tools.

```
                    LangGraph StateGraph
                    ====================

  +----------+     +-----------+     +----------+     +----------+
  |          |     |           |     |          |     |          |
  | SCANNER  |---->|  ANALYST  |---->| RISK MGR |---->| EXECUTOR |----> END
  |          |     |           |     |          |     |          |
  +----------+     +-----------+     +----------+     +----------+
       |                |                 |                |
       v                v                 v                v
  +---------+     +-----------+     +---------+     +---------+
  | Tools:  |     | Tools:    |     | Tools:  |     | Tools:  |
  | - market|     | - intra   |     | - get   |     | - place |
  |   status|     |   day     |     |   funds |     |   trade |
  | - prices|     | - daily   |     | - get   |     | - get   |
  | - scan  |     | - save    |     |   posns |     |   price |
  | - nifty |     |   signal  |     | - check |     | - save  |
  |   50    |     | - quote   |     |   loss  |     |   signal|
  +---------+     +-----------+     +---------+     +---------+
```

### 2.2 State Definition

```python
class TradingState(TypedDict):
    messages:          Annotated[list, add_messages]  # Conversation history
    scanner_output:    str    # Scanner agent's findings
    analyst_output:    str    # Analyst's technical analysis
    risk_output:       str    # Risk manager's decisions
    execution_output:  str    # Executor's trade confirmations
    portfolio_output:  str    # Portfolio review summary
    current_phase:     str    # Current pipeline stage
    trade_decisions:   list   # Approved trades for execution
    error:             str    # Error messages (if any)
```

### 2.3 Agent Specifications

#### Agent 1: Market Scanner

| Property | Value |
|----------|-------|
| **Mission** | Find stocks to BUY right now from NIFTY 50 |
| **Model** | GPT-4o |
| **Max Iterations** | 15 |
| **Output** | Top 3 BUY candidates with reasoning |

**Algorithm:**
```
1. check_market_status()           -- Verify market is open
2. get_nifty50_stocks()            -- Get full NIFTY 50 list
3. get_multiple_stock_prices()     -- Fetch prices for top 10 stocks
4. run_intraday_analysis()         -- 5-min candle analysis on movers
5. Select top 3 stocks with strongest BUY signals
6. Output: Symbol, price, signal strength, reasoning
```

#### Agent 2: Technical Analyst

| Property | Value |
|----------|-------|
| **Mission** | Generate BUY signals with exact entry, stop-loss, and target |
| **Model** | GPT-4o |
| **Max Iterations** | 15 |
| **Output** | Entry price, SL, target, strength for each candidate |

**Algorithm:**
```
1. For each scanner candidate:
   a. run_technical_analysis()      -- Daily timeframe (SMA, EMA, RSI, MACD...)
   b. run_intraday_analysis()       -- 5-min candle analysis
   c. get_stock_price()             -- Current price verification
   d. Compute entry = current price
   e. Compute SL = entry - (ATR * 1.5) or entry * 0.985
   f. Compute target = entry + (risk * 2.0) or entry * 1.03
   g. save_signal()                 -- Persist to database
2. Output ranked list with all parameters
```

#### Agent 3: Risk Manager

| Property | Value |
|----------|-------|
| **Mission** | Approve every valid trade with correct position sizing |
| **Model** | GPT-4o |
| **Max Iterations** | 10 |
| **Output** | APPROVED / REJECTED for each trade with position size |

**Risk Validation Checks:**
```
1. get_available_funds()            -- Check available capital
2. get_current_positions()          -- Count open positions (max 3)
3. check_daily_loss()               -- Verify daily loss < 3%
4. For each trade:
   a. Position size = min(capital * 10%, available_funds / 3)
   b. Quantity = floor(position_size / entry_price)
   c. Risk per trade = (entry - SL) * quantity
   d. Reward per trade = (target - entry) * quantity
   e. Risk:Reward ratio >= 1:1.5 ? APPROVE : REJECT
5. Output: APPROVED trades with exact quantity
```

#### Agent 4: Trade Executor

| Property | Value |
|----------|-------|
| **Mission** | Execute every approved trade immediately via Groww API |
| **Model** | GPT-4o |
| **Max Iterations** | 10 |
| **Output** | Order IDs and fill confirmations |

**Execution Flow:**
```
1. For each APPROVED trade:
   a. get_stock_price()             -- Final price check
   b. place_trade(                  -- Groww API call
        symbol, "BUY", quantity,
        order_type="MARKET",
        product="MIS"
      )
   c. Verify order status (EXECUTED / PLACED)
   d. Auto-register with PositionMonitor
   e. save_signal()                 -- Record execution
2. Output: Execution summary with order IDs and fill prices
```

---

<br>

## 3. Technical Analysis Engine

### 3.1 Indicator Computation Pipeline

```
                   Raw OHLCV Data (5-day, 5-min candles)
                                |
                                v
            +-------------------+-------------------+
            |                   |                   |
            v                   v                   v
     +------------+     +-------------+     +-----------+
     |   Trend    |     |  Momentum   |     | Volatility|
     | Indicators |     | Indicators  |     | Indicators|
     +------------+     +-------------+     +-----------+
     | SMA 20/50  |     | RSI (14)    |     | Bollinger |
     | EMA 9/21   |     | Stochastic  |     | ATR       |
     | MACD       |     | ADX         |     | VWAP      |
     | EMA Cross  |     | OBV         |     |           |
     +-----+------+     +------+------+     +-----+-----+
            |                   |                   |
            v                   v                   v
     +------+------+     +------+------+     +------+------+
     | BUY / SELL  |     | BUY / SELL  |     | BUY / SELL  |
     | Signal      |     | Signal      |     | Signal      |
     +------+------+     +------+------+     +------+------+
            |                   |                   |
            +-------------------+-------------------+
                                |
                                v
                    +------------------------+
                    |  Candlestick Pattern   |
                    |  Recognition           |
                    |  (Doji, Hammer, Star,  |
                    |   Engulfing, Marubozu) |
                    +-----------+------------+
                                |
                                v
                    +------------------------+
                    |  COUNT-BASED MAJORITY  |
                    |  VOTING SYSTEM         |
                    |                        |
                    |  BUY count > SELL ?    |
                    |    --> Overall: BUY    |
                    |  SELL count > BUY ?    |
                    |    --> Overall: SELL   |
                    |  Equal ?              |
                    |    --> Compare scores  |
                    +-----------+------------+
                                |
                                v
                    +------------------------+
                    |   FINAL SIGNAL         |
                    |   Signal: BUY/SELL/HOLD|
                    |   Strength: 0.0 - 1.0  |
                    |   Entry / SL / Target  |
                    +------------------------+
```

### 3.2 Signal Voting Algorithm

The system uses a **count-based majority voting** approach instead of score-averaging to prevent NEUTRAL indicators from diluting strong signals:

```python
# Pseudocode for signal generation
buy_count  = count(indicators where signal == "BUY")
sell_count = count(indicators where signal == "SELL")

if buy_count > sell_count:
    overall = "BUY"
    strength = (buy_count / total) * avg_buy_strength * 1.5
elif sell_count > buy_count:
    overall = "SELL"
    strength = (sell_count / total) * avg_sell_strength * 1.5
else:
    # Tiebreaker: compare cumulative scores
    overall = "BUY" if buy_score > sell_score else "SELL"
```

### 3.3 Indicator Signal Zones

| Indicator | BUY Zone | SELL Zone | Neutral |
|-----------|----------|-----------|---------|
| **RSI** | < 30 (strong) / 30-45 (mild) | > 70 (strong) / 55-70 (mild) | 45-55 |
| **Stochastic** | K < 35 | K > 65 | 35-65 |
| **Bollinger %B** | < 0.3 | > 0.7 | 0.3-0.7 |
| **MACD** | Histogram rising + crossover | Histogram falling + crossover | No crossover |
| **EMA** | EMA9 > EMA21 (crossover) | EMA9 < EMA21 (crossover) | Converging |
| **VWAP** | Price > VWAP + deviation | Price < VWAP - deviation | Near VWAP |
| **ADX** | > 25 with +DI > -DI | > 25 with -DI > +DI | < 25 |

---

<br>

## 4. Broker Integration Layer

### 4.1 Groww API Integration

```
+------------------+      +-------------------+      +------------------+
|   Alpha-Prime    |      |    Groww API       |      |     NSE          |
|   Broker Module  |----->|    (REST)          |----->|   Exchange       |
|                  |      |                   |      |                  |
|  broker.py       |      |  JWT Auth         |      |  Order Matching  |
|  GrowwBroker     |      |  CASH Segment     |      |  Engine          |
+------------------+      +-------------------+      +------------------+
```

### 4.2 Order Flow

```
place_trade("SBIN", "BUY", 1)
        |
        v
+------------------+
| Build Order      |
| Params:          |
|  exchange: "NSE" |
|  segment: "CASH" |
|  product: "MIS"  |  <-- Critical: Must be "MIS" not "INTRADAY"
|  order_type:     |
|    "MARKET"      |
|  validity: "DAY" |
+--------+---------+
         |
         v
+------------------+
| Groww API Call    |
| POST /order      |
| Auth: JWT Token  |
+--------+---------+
         |
         v
+------------------+
| Response:        |
| groww_order_id   |
| order_status     |
| order_ref_id     |
+--------+---------+
         |
         v (1 second wait)
+------------------+
| Poll Status      |
| GET /order/status|
|                  |
| filled_quantity  |
| average_price    |  <-- Actual fill price
| order_status:    |
|   "EXECUTED"     |
+--------+---------+
         |
         v
+------------------+
| Register with    |
| PositionMonitor  |
| (trailing SL)    |
+------------------+
```

### 4.3 Key API Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `product` | `"MIS"` | Margin Intraday Square-off (auto-squared by exchange at 3:30) |
| `order_type` | `"MARKET"` | Immediate execution at best available price |
| `exchange` | `"NSE"` | National Stock Exchange of India |
| `segment` | `"CASH"` | Equity cash segment |
| `validity` | `"DAY"` | Valid for current trading day only |
| `transaction_type` | `"BUY"` / `"SELL"` | Trade direction |
| `trigger_price` | *(SL orders only)* | Only set for SL / SL_M order types |

---

<br>

## 5. Position Management System

### 5.1 PositionMonitor Architecture

```python
class PositionMonitor:
    """Thread-safe position monitoring with trailing stop-loss."""

    _peak_prices:  Dict[str, float]   # Highest price since entry
    _entry_prices: Dict[str, float]   # Original entry price
    _stop_losses:  Dict[str, float]   # Current trailing SL level
    _targets:      Dict[str, float]   # Target exit price
    _lock:         threading.Lock()    # Thread safety
```

### 5.2 Trailing Stop-Loss Mechanism

```
Price Movement Example (Entry: Rs. 100.00, Initial SL: Rs. 98.50)

    Rs. 104.00  .............*....peak
    Rs. 103.00  ..........*..     |
    Rs. 102.46  ..........       SL moves to Rs. 102.44 (1.5% below peak)
    Rs. 102.00  ........*..
    Rs. 101.00  ......*.
    Rs. 100.50  ....*..
    Rs. 100.00  --*-------- entry price
    Rs.  99.00  ..|........
    Rs.  98.50  --+-------- initial SL (1.5% below entry)

    Rules:
    1. SL starts at entry * (1 - 0.015) = Rs. 98.50
    2. When price rises to Rs. 104, SL moves to 104 * 0.985 = Rs. 102.44
    3. SL NEVER moves down -- only up
    4. If price drops below SL --> AUTO SELL (MARKET order)
```

### 5.3 Exit Decision Logic

```
Every 2 minutes during market hours:
    |
    +-- For each monitored position:
    |       |
    |       +-- Fetch current price
    |       |
    |       +-- Is price > peak_price?
    |       |       YES --> Update peak, recalculate trailing SL
    |       |       NO  --> Continue
    |       |
    |       +-- Is price <= trailing_stop_loss?
    |       |       YES --> SELL (stop-loss hit)
    |       |
    |       +-- Is price >= target_price?
    |       |       YES --> SELL (target reached)
    |       |
    |       +-- Is time >= 3:15 PM?
    |               YES --> SELL (auto square-off)
    |
    +-- END
```

---

<br>

## 6. Database Design

### 6.1 Entity-Relationship Diagram

```
+------------------+     +------------------+     +------------------+
|     trades       |     |     signals      |     |   agent_logs     |
+------------------+     +------------------+     +------------------+
| id (PK)          |     | id (PK)          |     | id (PK)          |
| timestamp        |     | timestamp        |     | timestamp        |
| symbol           |     | symbol           |     | agent_name       |
| action (BUY/SELL)|     | signal_type      |     | log_type         |
| quantity         |     | strength (0-1)   |     | message          |
| price            |     | entry_price      |     | tool_name        |
| order_type       |     | stop_loss        |     | tool_input (JSON)|
| order_id         |     | target           |     | tool_output(JSON)|
| status           |     | indicators (JSON)|     | error            |
| rationale        |     | reasoning        |     +------------------+
| pnl              |     | acted_on         |
+------------------+     +------------------+
                                                   +------------------+
+------------------+     +------------------+     |  system_state    |
| portfolio_       |     |   watchlist      |     +------------------+
|   snapshots      |     +------------------+     | key (PK)         |
+------------------+     | id (PK)          |     | value            |
| id (PK)          |     | symbol           |     | updated_at       |
| timestamp        |     | added_at         |     +------------------+
| total_capital    |     | notes            |
| invested_amount  |     +------------------+     +------------------+
| available_cash   |                              |  daily_summary   |
| total_value      |                              +------------------+
| daily_pnl        |                              | id (PK)          |
| positions (JSON) |                              | date             |
+------------------+                              | total_trades     |
                                                  | winning_trades   |
                                                  | total_pnl        |
                                                  | max_drawdown     |
                                                  | notes            |
                                                  +------------------+
```

### 6.2 Table Specifications

| Table | Records/Day | Retention | Purpose |
|-------|-------------|-----------|---------|
| `trades` | 5-15 | Permanent | Complete trade audit trail |
| `signals` | 20-50 | Permanent | AI signal history and analysis |
| `agent_logs` | 100-300 | 90 days | Debugging and monitoring |
| `portfolio_snapshots` | 10-20 | 30 days | Portfolio tracking over time |
| `daily_summary` | 1 | Permanent | Daily performance metrics |
| `watchlist` | N/A | User-managed | Custom stock tracking |
| `system_state` | N/A | Active | Key-value runtime config |

---

<br>

## 7. Scheduling & Automation

### 7.1 Daily Trading Timeline

```
  TIME         EVENT                          DESCRIPTION
  ====         =====                          ===========

  09:00 AM     Pre-Market Scan                Scan NIFTY 50 for potential movers
     |
  09:15 AM     Market Opens                   NSE trading session begins
     |
  09:20 AM     First Trading Cycle            Pipeline: Scan -> Analyze -> Risk -> Execute
     |
  09:22 AM     Position Monitor Starts        Check positions every 2 minutes
     |
     |         ... Trading cycles every 5-15 minutes ...
     |         ... Position monitor every 2 minutes ...
     |
  03:15 PM     Auto Square-Off                SELL all open MIS positions
     |
  03:25 PM     End-of-Day Review              Daily P&L summary and performance metrics
     |
  03:30 PM     Market Closes                  NSE trading session ends
     |
     |         System remains idle until next trading day
```

### 7.2 APScheduler Job Configuration

| Job | Trigger Type | Schedule | Market Hours Only |
|-----|-------------|----------|-------------------|
| `trading_cycle` | Interval | Every N minutes (configurable) | Yes |
| `position_monitor` | Interval | Every 2 minutes | Yes |
| `pre_market_scan` | Cron | 9:00 AM Mon-Fri | N/A |
| `auto_square_off` | Cron | 3:15 PM Mon-Fri | N/A |
| `eod_review` | Cron | 3:25 PM Mon-Fri | N/A |

---

<br>

## 8. Dashboard UI Design

### 8.1 Layout Structure

```
+-------------------+--------------------------------------------------+
|                   |                                                  |
|    SIDEBAR        |              MAIN CONTENT AREA                   |
|                   |                                                  |
|  +-------------+  |  +--------------------------------------------+ |
|  | API Keys    |  |  |  Tab Navigation:                           | |
|  | - OpenAI    |  |  |  [Portfolio] [Trades] [Signals] [Analysis] | |
|  | - Groww     |  |  |  [Agent Logs] [Watchlist]                  | |
|  +-------------+  |  +--------------------------------------------+ |
|                   |                                                  |
|  +-------------+  |  +--------------------------------------------+ |
|  | Trading     |  |  |                                            | |
|  | Mode:       |  |  |  Tab Content                               | |
|  | [Paper/Live]|  |  |  (varies by selected tab)                  | |
|  +-------------+  |  |                                            | |
|                   |  |  - Portfolio: P&L cards + positions table   | |
|  +-------------+  |  |  - Trades: trade history dataframe         | |
|  | Risk Params |  |  |  - Signals: signal cards + strength bars   | |
|  | - Capital   |  |  |  - Analysis: candlestick charts (Plotly)   | |
|  | - Max Loss  |  |  |  - Logs: color-coded agent activities      | |
|  | - Position% |  |  |  - Watchlist: custom stock tracker         | |
|  +-------------+  |  |                                            | |
|                   |  +--------------------------------------------+ |
|  +-------------+  |                                                  |
|  | Controls    |  |                                                  |
|  | [Start]     |  |                                                  |
|  | [Stop]      |  |                                                  |
|  | [Run Now]   |  |                                                  |
|  | [Square Off]|  |                                                  |
|  +-------------+  |                                                  |
|                   |                                                  |
|  +-------------+  |                                                  |
|  | Position    |  |                                                  |
|  | Monitor     |  |                                                  |
|  | Status      |  |                                                  |
|  +-------------+  |                                                  |
|                   |                                                  |
+-------------------+--------------------------------------------------+
```

### 8.2 Key UI Components

| Component | Library | Refresh Rate | Description |
|-----------|---------|-------------|-------------|
| P&L Metrics | `st.metric` | 30 sec | Total P&L, invested, available cash |
| Positions Table | `st.dataframe` | 30 sec | Open positions with live prices |
| Trade History | `st.dataframe` | On demand | Sortable/filterable trade log |
| Candlestick Chart | `plotly.graph_objects` | On demand | Interactive OHLCV + volume |
| RSI Gauge | `plotly.graph_objects` | On demand | Current RSI with zones |
| Agent Logs | `st.expander` | 30 sec | Color-coded by agent type |
| SL Status | `st.progress` | 30 sec | Visual trailing SL indicator |

---

<br>

## 9. Tool System Design

### 9.1 LangChain Tool Registry

All tools are defined as `@tool` decorated functions in `tools.py` and bound to agents via LangGraph:

| # | Tool Name | Used By | Description |
|---|-----------|---------|-------------|
| 1 | `check_market_status` | Scanner | Returns whether NSE market is currently open |
| 2 | `get_nifty50_stocks` | Scanner | Returns list of NIFTY 50 stock symbols |
| 3 | `get_stock_price` | All | Fetches current price, day change, volume for a symbol |
| 4 | `get_multiple_stock_prices` | Scanner | Batch price fetch for multiple symbols |
| 5 | `run_technical_analysis` | Analyst | Daily timeframe analysis (9+ indicators) |
| 6 | `run_intraday_analysis` | Scanner, Analyst | 5-min candle analysis with pattern detection |
| 7 | `save_signal` | Analyst, Executor | Persist signal to database |
| 8 | `get_available_funds` | Risk Mgr | Fetch available margin from Groww |
| 9 | `get_current_positions` | Risk Mgr | Fetch open MIS positions |
| 10 | `check_daily_loss` | Risk Mgr | Calculate day's realized + unrealized P&L |
| 11 | `place_trade` | Executor | Place BUY/SELL order via Groww API |
| 12 | `get_portfolio_summary` | All | Overall portfolio state |
| 13 | `get_trade_history` | Executor | Recent trades from database |
| 14 | `get_holdings` | Risk Mgr | Delivery holdings from Groww |

### 9.2 Tool-Agent Binding

```python
# Scanner gets market scanning tools
scanner_tools = [check_market_status, get_nifty50_stocks,
                 get_multiple_stock_prices, run_intraday_analysis, save_signal]

# Analyst gets deep analysis tools
analyst_tools = [run_intraday_analysis, run_technical_analysis,
                 get_stock_price, save_signal]

# Risk Manager gets portfolio/risk tools
risk_tools = [get_available_funds, get_current_positions,
              check_daily_loss, get_stock_price, get_portfolio_summary]

# Executor gets trading tools
exec_tools = [place_trade, get_stock_price, save_signal, get_trade_history]
```

---

<br>

## 10. Configuration & Security

### 10.1 Environment Configuration

```env
# ---- API Keys ----
OPENAI_API_KEY=sk-...              # GPT-4o access
GROWW_API_KEY=eyJ...               # Groww JWT token (rotated daily)

# ---- Trading Parameters ----
TRADING_MODE=live                   # paper | live
TRADING_CAPITAL=5000                # Capital in INR
MAX_DAILY_LOSS_PCT=3.0             # Max daily loss percentage
MAX_POSITION_SIZE_PCT=10.0         # Max % of capital per trade
MAX_OPEN_POSITIONS=3               # Max concurrent positions
DEFAULT_STOP_LOSS_PCT=1.5          # Trailing SL percentage
DEFAULT_TARGET_PCT=3.0             # Target profit percentage
ANALYSIS_INTERVAL_MINUTES=5        # Pipeline frequency
```

### 10.2 Security Measures

| Measure | Implementation |
|---------|---------------|
| API Key Storage | `.env` file (excluded from git via `.gitignore`) |
| Credential Logging | API keys are **never** written to logs |
| Network Security | All external API calls use HTTPS |
| Input Validation | Pydantic validators on all config fields |
| JWT Token Handling | Raw string passed to Groww SDK (not dict) |

---

<br>

## 11. Error Handling & Resilience

### 11.1 Failure Modes & Recovery

| Failure | Impact | Recovery Strategy |
|---------|--------|-------------------|
| OpenAI API timeout | Pipeline stalls | 120s timeout + retry with exponential backoff |
| Groww API auth failure | Cannot place orders | Log error, skip cycle, alert on dashboard |
| Groww order rejection | Trade not placed | Log rejection reason, continue with next trade |
| yfinance data unavailable | No market data | Use cached data (5-min TTL), skip if stale |
| Database connection error | No persistence | Auto-reconnect with SQLAlchemy pool |
| Network disconnection | All APIs fail | Graceful degradation, resume on reconnect |
| Position Monitor crash | SL not enforced | Separate thread with exception handling + restart |
| Invalid stock symbol | Analysis fails | Symbol normalization + validation |

### 11.2 Graceful Degradation

```
Priority of operations during partial failure:

1. HIGHEST: Position monitoring + trailing SL enforcement
2. HIGH:    Auto square-off at 3:15 PM
3. MEDIUM:  Trade execution for approved signals
4. LOW:     New market scanning and analysis
5. LOWEST:  Dashboard updates and logging
```

---

<br>

## 12. AWS Deployment Architecture

### 12.1 Current Deployment (Local)

```
+-----------------------------------+
|        Developer Machine          |
|                                   |
|   Python 3.13 + venv              |
|   Streamlit (localhost:8501)      |
|   SQLite (local file)            |
|   APScheduler (in-process)       |
+-----------------------------------+
```

### 12.2 Production AWS Architecture (Planned)

```
+-------------------------------------------------------------------+
|                        AWS Cloud                                   |
|                                                                   |
|  +------------------+     +-------------------+                   |
|  |  Amazon ECS /    |     |  Amazon RDS       |                   |
|  |  EC2 (t3.medium) |     |  (PostgreSQL)     |                   |
|  |                  |     |                   |                   |
|  |  Docker Container|     |  Replaces SQLite  |                   |
|  |  Alpha-Prime App |---->|  for production   |                   |
|  |  + Streamlit     |     |  scale            |                   |
|  +--------+---------+     +-------------------+                   |
|           |                                                       |
|           v                                                       |
|  +------------------+     +-------------------+                   |
|  |  Amazon Bedrock  |     |  AWS CloudWatch   |                   |
|  |  (Future)        |     |  Monitoring &     |                   |
|  |                  |     |  Alerting         |                   |
|  |  Replace OpenAI  |     |                   |                   |
|  |  with Claude /   |     |  Trade alerts     |                   |
|  |  Titan models    |     |  System health    |                   |
|  +------------------+     +-------------------+                   |
|                                                                   |
|  +------------------+     +-------------------+                   |
|  |  AWS Lambda      |     |  Amazon SNS /     |                   |
|  |  (Future)        |     |  SES (Future)     |                   |
|  |                  |     |                   |                   |
|  |  Serverless      |     |  Trade            |                   |
|  |  trading cycles  |     |  notifications    |                   |
|  +------------------+     |  via SMS/Email    |                   |
|                           +-------------------+                   |
|                                                                   |
|  +------------------+                                             |
|  |  Amazon          |                                             |
|  |  SageMaker       |                                             |
|  |  (Future)        |                                             |
|  |                  |                                             |
|  |  Custom ML       |                                             |
|  |  signal models   |                                             |
|  +------------------+                                             |
+-------------------------------------------------------------------+
```

### 12.3 Scaling Strategy

| Phase | Users | Infrastructure | Est. Cost/Month |
|-------|-------|---------------|-----------------|
| **Phase 1** (Current) | 1 | Local machine | Rs. 3,000 (OpenAI only) |
| **Phase 2** | 1 | AWS EC2 + RDS | Rs. 6,000 - 8,000 |
| **Phase 3** | 10 | ECS + Bedrock + RDS | Rs. 15,000 - 25,000 |
| **Phase 4** | 100+ | Multi-tenant SaaS | Rs. 50,000+ |

---

<br>

<div align="center">

---

**Alpha-Prime** | System Design Document

Built for **AWS AI for Bharat Hackathon 2025**

*Empowering Indian retail investors with AI-driven autonomous trading*

---

</div>
