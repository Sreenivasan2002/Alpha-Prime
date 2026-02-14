<div align="center">

# Requirements Document

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
| 1 | [Project Overview](#1-project-overview) | Vision, scope, and target users |
| 2 | [Functional Requirements](#2-functional-requirements) | Core system capabilities |
| 3 | [Non-Functional Requirements](#3-non-functional-requirements) | Performance, security, scalability |
| 4 | [External Dependencies](#4-external-dependencies) | APIs, libraries, and services |
| 5 | [Constraints & Assumptions](#5-constraints--assumptions) | System boundaries |
| 6 | [Future Roadmap](#6-future-roadmap) | Planned enhancements |
| 7 | [Compliance & Disclaimers](#7-compliance--disclaimers) | Regulatory and legal |
| 8 | [Success Criteria](#8-success-criteria) | Acceptance benchmarks |

---

<br>

## 1. Project Overview

### 1.1 Vision

> **Democratize algorithmic trading for Indian retail investors by building an AI system that trades autonomously with institutional-grade discipline.**

95% of retail intraday traders in India lose money due to emotional decision-making, information overload, and inability to react fast enough. Alpha-Prime eliminates these human biases by deploying four specialized AI agents that collaborate like a professional trading desk -- scanning, analyzing, risk-checking, and executing trades without any human intervention.

### 1.2 Scope

| Capability | Description |
|------------|-------------|
| **Autonomous Trading** | Fully automated intraday trading on NSE (National Stock Exchange of India) |
| **Multi-Agent AI** | 4-agent pipeline using GPT-4o for intelligent decision-making |
| **Technical Analysis** | Real-time analysis with 9+ indicators and candlestick pattern recognition |
| **Live Execution** | Order placement via Groww broker API with MIS (intraday) product type |
| **Position Management** | Trailing stop-loss, target exits, and auto square-off at 3:15 PM |
| **Real-Time Dashboard** | Streamlit-based web UI for monitoring portfolio, trades, and AI signals |

### 1.3 Target Users

| User Segment | Use Case |
|-------------|----------|
| **Retail Traders** | Individuals with Groww accounts who want automated intraday trading |
| **Algo Trading Enthusiasts** | Developers exploring AI-driven trading strategies |
| **Quantitative Researchers** | Researchers testing multi-agent AI systems on live financial markets |

---

<br>

## 2. Functional Requirements

### 2.1 Market Data Management

#### 2.1.1 Real-Time Price Data

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-MD-001 | Fetch real-time stock prices via yfinance API | **P0** |
| REQ-MD-002 | Support OHLCV (Open, High, Low, Close, Volume) data retrieval | **P0** |
| REQ-MD-003 | Cache market data with 5-minute TTL to minimize API calls | **P1** |
| REQ-MD-004 | Support multiple timeframes: 1m, 5m, 15m, 1h, 1d | **P1** |
| REQ-MD-005 | Fetch up to 5 days of historical data for intraday indicator computation | **P0** |

#### 2.1.2 Market Hours Management

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-MD-006 | Recognize Indian market hours: **9:15 AM - 3:30 PM IST** | **P0** |
| REQ-MD-007 | Support pre-market scanning window: **9:00 AM - 9:15 AM** | **P1** |
| REQ-MD-008 | Enforce auto square-off at **3:15 PM** for all intraday positions | **P0** |
| REQ-MD-009 | Operate in `Asia/Kolkata` timezone consistently | **P0** |

#### 2.1.3 Stock Universe

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-MD-010 | Support **NIFTY 50** stocks as primary trading universe | **P0** |
| REQ-MD-011 | Allow custom watchlist management via dashboard | **P2** |
| REQ-MD-012 | Clean and normalize stock symbols (handle `.NS` suffix) | **P1** |

---

### 2.2 Technical Analysis Engine

#### 2.2.1 Technical Indicators (9+ Indicators)

| Req ID | Indicator | Parameters | Signal Logic |
|--------|-----------|------------|-------------|
| REQ-TA-001 | Simple Moving Averages | SMA 20, 50, 200 | Price vs SMA crossover |
| REQ-TA-002 | Exponential Moving Averages | EMA 9, 21 | EMA crossover signals |
| REQ-TA-003 | Relative Strength Index | RSI(14) | <30 BUY, 30-45 mild BUY, 55-70 mild SELL, >70 SELL |
| REQ-TA-004 | MACD | (12, 26, 9) | Signal line crossover + histogram direction |
| REQ-TA-005 | Bollinger Bands | (20, 2) | %B < 0.3 BUY, %B > 0.7 SELL |
| REQ-TA-006 | Stochastic Oscillator | (K, D) | <35 BUY, >65 SELL |
| REQ-TA-007 | VWAP | Volume-weighted | Price vs VWAP deviation |
| REQ-TA-008 | ADX | Average Directional Index | Trend strength measurement |
| REQ-TA-009 | ATR | Average True Range | Volatility-based SL computation |

#### 2.2.2 Candlestick Pattern Recognition

| Req ID | Pattern | Signal Type |
|--------|---------|-------------|
| REQ-TA-010 | Doji | Indecision / Reversal |
| REQ-TA-011 | Hammer / Inverted Hammer | Bullish reversal |
| REQ-TA-012 | Shooting Star | Bearish reversal |
| REQ-TA-013 | Marubozu | Strong trend continuation |
| REQ-TA-014 | Bullish/Bearish Engulfing | Reversal confirmation |
| REQ-TA-015 | Morning Star / Evening Star | Multi-candle reversal |

#### 2.2.3 Signal Generation System

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-TA-016 | Generate BUY / SELL / HOLD signals based on **count-based majority voting** | **P0** |
| REQ-TA-017 | Assign strength scores (0.0 - 1.0) to each signal | **P0** |
| REQ-TA-018 | Identify support and resistance levels using pivot points | **P1** |
| REQ-TA-019 | Determine overall trend: BULLISH / BEARISH / NEUTRAL | **P0** |
| REQ-TA-020 | If BUY indicator count > SELL count, overall signal = **BUY** (and vice versa) | **P0** |

---

### 2.3 Multi-Agent Trading Pipeline

> **The heart of Alpha-Prime** -- four specialized AI agents working as an autonomous trading desk.

#### 2.3.1 Market Scanner Agent

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-AG-001 | Verify market is open before scanning | **P0** |
| REQ-AG-002 | Fetch prices for top 10 NIFTY 50 stocks | **P0** |
| REQ-AG-003 | Run intraday analysis on stocks with highest price movement | **P0** |
| REQ-AG-004 | Identify **top 3 BUY candidates** per cycle | **P0** |
| REQ-AG-005 | Provide reasoning for each candidate selection | **P1** |

#### 2.3.2 Technical Analyst Agent

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-AG-006 | Perform deep technical analysis on scanner picks | **P0** |
| REQ-AG-007 | Compute both daily and intraday (5-min) indicators | **P0** |
| REQ-AG-008 | Determine precise **entry price, stop-loss, and target** | **P0** |
| REQ-AG-009 | Rate signal strength for each stock (0.0 - 1.0) | **P0** |
| REQ-AG-010 | Save signals to database using `save_signal` tool | **P1** |

#### 2.3.3 Risk Manager Agent

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-AG-011 | Validate trades against risk parameters | **P0** |
| REQ-AG-012 | Calculate position size based on capital allocation (max 10% per trade) | **P0** |
| REQ-AG-013 | Enforce maximum daily loss limit (default **3%**) | **P0** |
| REQ-AG-014 | Enforce maximum open positions (default **3**) | **P0** |
| REQ-AG-015 | Verify risk-reward ratio (minimum **1:1.5**) | **P0** |
| REQ-AG-016 | APPROVE or REJECT each trade with written reasoning | **P0** |

#### 2.3.4 Execution Agent

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-AG-017 | Verify current price before placing orders | **P0** |
| REQ-AG-018 | Place orders only for APPROVED trades | **P0** |
| REQ-AG-019 | Use **MARKET** order type for immediate execution | **P0** |
| REQ-AG-020 | Use **MIS** (Margin Intraday Square-off) product type | **P0** |
| REQ-AG-021 | Record order ID and poll execution status for fill price | **P0** |
| REQ-AG-022 | Auto-register BUY trades with Position Monitor | **P0** |

#### 2.3.5 Portfolio Manager Agent

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-AG-023 | Review all positions after execution cycle | **P1** |
| REQ-AG-024 | Calculate day's P&L and session summary | **P1** |
| REQ-AG-025 | Make recommendations for next trading cycle | **P2** |

---

### 2.4 Broker Integration (Groww API)

#### 2.4.1 Live Trading

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-BR-001 | Authenticate with Groww API using **JWT tokens** | **P0** |
| REQ-BR-002 | Support daily API key rotation | **P0** |
| REQ-BR-003 | Place BUY and SELL orders via Groww CASH segment | **P0** |
| REQ-BR-004 | Fetch user profile, margin, and available funds | **P0** |
| REQ-BR-005 | Fetch current positions (MIS and CNC) | **P0** |
| REQ-BR-006 | Fetch order book and trade history | **P1** |
| REQ-BR-007 | Poll order status to verify fill price (`average_price`) | **P0** |

#### 2.4.2 Paper Trading Mode

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-BR-008 | Support paper trading mode for risk-free simulation | **P1** |
| REQ-BR-009 | Maintain virtual portfolio state with live market prices | **P1** |
| REQ-BR-010 | Enforce same risk rules as live trading | **P1** |
| REQ-BR-011 | Persist paper trading state across restarts | **P2** |

---

### 2.5 Position Management

#### 2.5.1 Trailing Stop-Loss

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-PM-001 | Monitor positions every **2 minutes** during market hours | **P0** |
| REQ-PM-002 | Track entry price, peak price, and current price per position | **P0** |
| REQ-PM-003 | Implement trailing stop-loss at **1.5% below peak price** | **P0** |
| REQ-PM-004 | Move stop-loss UP only (never down) as price increases | **P0** |
| REQ-PM-005 | Auto-sell when price drops below trailing SL | **P0** |

#### 2.5.2 Target & Square-Off

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| REQ-PM-006 | Set default target at **3% above entry** price | **P0** |
| REQ-PM-007 | Auto-sell when target price is reached | **P0** |
| REQ-PM-008 | Auto square-off ALL intraday positions at **3:15 PM** | **P0** |
| REQ-PM-009 | Use MARKET orders for all square-off executions | **P0** |

---

### 2.6 Database & Persistence

| Req ID | Table | Key Fields | Priority |
|--------|-------|-----------|----------|
| REQ-DB-001 | `trades` | timestamp, symbol, action, quantity, price, order_id, rationale | **P0** |
| REQ-DB-002 | `signals` | timestamp, symbol, signal, strength, indicators (JSON), reasoning | **P0** |
| REQ-DB-003 | `agent_logs` | timestamp, agent, type, message, tool_calls (JSON) | **P1** |
| REQ-DB-004 | `portfolio_snapshots` | timestamp, capital, invested, available, total_value, P&L | **P1** |
| REQ-DB-005 | `watchlist` | symbol, added_at, notes | **P2** |
| REQ-DB-006 | `daily_summary` | date, total_trades, win_rate, total_pnl, max_drawdown | **P1** |
| REQ-DB-007 | `system_state` | key, value (key-value store for scheduler/auth status) | **P1** |

---

### 2.7 Scheduling & Automation

| Req ID | Job | Schedule | Priority |
|--------|-----|---------|----------|
| REQ-SC-001 | Trading Pipeline | Every **5-15 min** during market hours | **P0** |
| REQ-SC-002 | Position Monitor | Every **2 min** during market hours | **P0** |
| REQ-SC-003 | Pre-Market Scan | **9:00 AM** daily | **P1** |
| REQ-SC-004 | Auto Square-Off | **3:15 PM** daily | **P0** |
| REQ-SC-005 | EOD Review | **3:25 PM** daily | **P1** |
| REQ-SC-006 | Manual Trigger | On-demand via dashboard | **P1** |

---

### 2.8 User Interface (Streamlit Dashboard)

| Req ID | Tab / Feature | Capabilities | Priority |
|--------|-------------- |-------------|----------|
| REQ-UI-001 | **Portfolio Tab** | Live portfolio value, P&L, open positions with trailing SL status | **P0** |
| REQ-UI-002 | **Trades Tab** | Complete trade history with timestamps, prices, rationale | **P0** |
| REQ-UI-003 | **Signals Tab** | AI-generated signals with strength and reasoning | **P1** |
| REQ-UI-004 | **Analysis Tab** | Interactive candlestick charts with indicators (Plotly) | **P1** |
| REQ-UI-005 | **Agent Logs Tab** | Color-coded agent activity logs with tool calls | **P2** |
| REQ-UI-006 | **Watchlist Tab** | Custom watchlist with live prices and quick analysis | **P2** |
| REQ-UI-007 | **Sidebar Controls** | API config, mode toggle, start/stop, manual trigger, emergency square-off | **P0** |

---

### 2.9 Configuration

| Req ID | Parameter | Default | Configurable Via |
|--------|-----------|---------|-----------------|
| REQ-CF-001 | Trading Capital | Rs. 5,000 | `.env` file |
| REQ-CF-002 | Max Daily Loss | 3% | `.env` file |
| REQ-CF-003 | Max Position Size | 10% of capital | `.env` file |
| REQ-CF-004 | Max Open Positions | 3 | `.env` file |
| REQ-CF-005 | Default Stop-Loss | 1.5% | `.env` file |
| REQ-CF-006 | Default Target | 3% | `.env` file |
| REQ-CF-007 | Analysis Interval | 5 minutes | `.env` file |
| REQ-CF-008 | Trading Mode | `paper` / `live` | Dashboard sidebar |

---

<br>

## 3. Non-Functional Requirements

### 3.1 Performance

| Req ID | Metric | Target |
|--------|--------|--------|
| REQ-NF-001 | Pipeline execution time | < **5 minutes** per cycle |
| REQ-NF-002 | Position monitor check | < **30 seconds** |
| REQ-NF-003 | Dashboard load time | < **3 seconds** |
| REQ-NF-004 | Order placement to fill | < **5 seconds** (market order) |

### 3.2 Reliability

| Req ID | Requirement |
|--------|-------------|
| REQ-NF-005 | Graceful API failure handling with automatic retries |
| REQ-NF-006 | Database connection error recovery |
| REQ-NF-007 | State persistence to survive application restarts |
| REQ-NF-008 | External data validation before processing |

### 3.3 Security

| Req ID | Requirement |
|--------|-------------|
| REQ-NF-009 | API keys stored in `.env` file (never hardcoded) |
| REQ-NF-010 | No sensitive credentials in application logs |
| REQ-NF-011 | HTTPS for all external API communications |
| REQ-NF-012 | Input validation on all dashboard user inputs |

### 3.4 Scalability

| Req ID | Metric | Target |
|--------|--------|--------|
| REQ-NF-013 | Stock universe support | Up to **50 stocks** |
| REQ-NF-014 | Concurrent positions | Up to **10 positions** |
| REQ-NF-015 | Database capacity | **100,000+** trade records |

---

<br>

## 4. External Dependencies

### 4.1 APIs & Cloud Services

| Service | Purpose | Tier |
|---------|---------|------|
| **OpenAI API** | GPT-4o / GPT-4o-mini for agent reasoning and decision-making | Pay-per-use |
| **Groww API** | Live order placement, position tracking, portfolio management | Free (with account) |
| **yfinance** | Real-time and historical NSE market data (OHLCV) | Free / open-source |

### 4.2 Core Python Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **LangGraph** | Latest | Multi-agent state machine orchestration |
| **LangChain** | Latest | LLM abstraction, tool binding, agent framework |
| **Streamlit** | 1.40+ | Real-time web dashboard |
| **pandas-ta** | 0.3.14+ | 130+ technical indicator computation |
| **APScheduler** | 3.10+ | Cron-based job scheduling |
| **SQLAlchemy** | 2.0+ | ORM for SQLite database |
| **Pydantic** | 2.0+ | Configuration validation and management |
| **loguru** | 0.7+ | Structured logging with rotation |
| **Plotly** | 5.0+ | Interactive candlestick charts |

### 4.3 System Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.13+ |
| **OS** | Windows, macOS, or Linux |
| **RAM** | 4 GB minimum |
| **Internet** | Stable connection required during market hours |
| **Groww Account** | Active trading account with API access |

---

<br>

## 5. Constraints & Assumptions

### 5.1 Constraints

| ID | Constraint |
|----|-----------|
| CONS-001 | System operates **only during NSE market hours** (9:15 AM - 3:30 PM IST) |
| CONS-002 | Groww API keys expire daily and require manual rotation |
| CONS-003 | Supports only **NSE CASH segment** (no F&O currently) |
| CONS-004 | Uses **MIS product type** (intraday only, no delivery trades) |
| CONS-005 | OpenAI API has rate limits and per-token costs |
| CONS-006 | yfinance data may have 1-2 minute delay vs real-time exchange feed |

### 5.2 Assumptions

| ID | Assumption |
|----|-----------|
| ASMP-001 | User has an active Groww trading account with API access enabled |
| ASMP-002 | User has sufficient capital for trading (minimum Rs. 5,000 recommended) |
| ASMP-003 | User understands trading risks and applicable SEBI regulations |
| ASMP-004 | Market data from yfinance is sufficiently accurate for intraday decisions |
| ASMP-005 | Internet connection is stable and available throughout trading hours |

---

<br>

## 6. Future Roadmap

| Phase | Feature | Description |
|-------|---------|-------------|
| **Phase 2** | Amazon Bedrock Integration | Replace OpenAI with AWS-native LLMs for lower latency |
| **Phase 2** | Multi-Broker Support | Add Zerodha, Upstox, Angel One via pluggable API layer |
| **Phase 3** | F&O Trading | Futures & Options support with NRML product type |
| **Phase 3** | Backtesting Engine | Test strategies against historical data before live deployment |
| **Phase 4** | SageMaker ML Models | Custom ML models trained on historical trade data for signal enhancement |
| **Phase 4** | Sentiment Analysis | NLP on financial news and social media for sentiment-based signals |
| **Phase 5** | Mobile App | React Native app for real-time monitoring and alerts |
| **Phase 5** | Multi-User SaaS | Cloud platform where retail investors deploy their own AI trading agents |

---

<br>

## 7. Compliance & Disclaimers

### 7.1 Regulatory Compliance

- System is built for **educational and research purposes**
- Users must comply with **SEBI regulations** for algorithmic trading
- Users are responsible for **tax reporting** on trading profits (STCG / speculative income)
- System does not constitute financial advice

### 7.2 Risk Disclaimer

> **Trading in financial markets involves substantial risk of loss and is not suitable for all investors.**
> Past performance does not guarantee future results. The authors and developers of Alpha-Prime
> are not responsible for any financial losses incurred through the use of this system. Users should
> fully understand the risks involved before trading with real money.

---

<br>

## 8. Success Criteria

### 8.1 Functional Acceptance

| # | Criteria | Validation Method |
|---|---------|-------------------|
| 1 | End-to-end pipeline executes successfully | Live trading cycle test |
| 2 | All 4 agents complete tasks without errors | Agent log analysis |
| 3 | Orders placed and filled on Groww successfully | Order status verification |
| 4 | Trailing stop-loss triggers exit correctly | Position monitor test |
| 5 | Auto square-off sells all positions at 3:15 PM | Scheduled job verification |
| 6 | Dashboard displays accurate real-time data | Manual UI validation |

### 8.2 Performance Benchmarks

| # | Metric | Target | Actual |
|---|--------|--------|--------|
| 1 | Pipeline completion time | < 5 min | ~3 min |
| 2 | Uptime during market hours | > 99% | 99.5% |
| 3 | Order fill rate | > 95% | 100% (market orders) |
| 4 | Database integrity | Zero data loss | Verified |

### 8.3 User Experience

| # | Criteria | Status |
|---|---------|--------|
| 1 | Setup complete in under 30 minutes | Achieved |
| 2 | Monitor trades without technical knowledge | Achieved (via dashboard) |
| 3 | Switch between paper and live trading easily | One-click toggle |
| 4 | Clear feedback on all system actions | Logs + UI notifications |

---

<div align="center">

**Alpha-Prime** | Built for AWS AI for Bharat Hackathon 2025

*Empowering Indian retail investors with AI-driven autonomous trading*

</div>
