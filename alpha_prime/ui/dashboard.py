"""
Streamlit Dashboard for Alpha-Prime Trading System
Beautiful, readable UI with real-time portfolio tracking and controls.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pytz
import json
import sys
import os
import traceback
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from alpha_prime.core.config import settings
from alpha_prime.core.database import (
    get_trades, get_signals, get_agent_logs, get_portfolio_history,
    get_watchlist, add_to_watchlist, remove_from_watchlist,
    get_open_positions, init_database, get_system_state
)
from alpha_prime.core.broker import get_broker, reset_broker, GrowwBroker
from alpha_prime.core.scheduler import trading_scheduler, position_monitor
from alpha_prime.data.market_data import (
    market_data, is_market_open, is_pre_market, time_to_market_open, clean_symbol
)
from alpha_prime.data.technical_analysis import technical_analyzer

IST = pytz.timezone("Asia/Kolkata")

# ---- Page Config ----
st.set_page_config(
    page_title="Alpha-Prime | Autonomous Trading",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Improved CSS ----
st.markdown("""
<style>
    /* Global font improvements */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Header styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }
    .sub-title {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 0;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        white-space: nowrap;
    }
    .badge-green { background: #064e3b; color: #34d399; border: 1px solid #34d399; }
    .badge-red { background: #450a0a; color: #f87171; border: 1px solid #f87171; }
    .badge-yellow { background: #422006; color: #fbbf24; border: 1px solid #fbbf24; }
    .badge-blue { background: #1e3a5f; color: #60a5fa; border: 1px solid #60a5fa; }

    /* Metric cards - prevent truncation */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px 14px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        overflow: visible !important;
    }
    div[data-testid="stMetric"] label {
        color: #9ca3af !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    /* Card container */
    .info-card {
        background: #1a1a2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }

    /* Status header bar */
    .status-bar {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .status-item {
        text-align: center;
    }
    .status-label {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .status-value {
        font-size: 1rem;
        font-weight: 600;
    }

    /* Agent log styling */
    .log-entry {
        font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace;
        font-size: 0.78rem;
        padding: 4px 8px;
        border-left: 3px solid transparent;
        margin: 2px 0;
        border-radius: 0 4px 4px 0;
        line-height: 1.5;
    }
    .log-scanner { border-left-color: #60a5fa; background: rgba(96,165,250,0.05); }
    .log-analyst { border-left-color: #a78bfa; background: rgba(167,139,250,0.05); }
    .log-risk { border-left-color: #fbbf24; background: rgba(251,191,36,0.05); }
    .log-execution { border-left-color: #34d399; background: rgba(52,211,153,0.05); }
    .log-error { border-left-color: #f87171; background: rgba(248,113,113,0.08); }
    .log-trade { border-left-color: #f472b6; background: rgba(244,114,182,0.05); }
    .log-default { border-left-color: #64748b; background: rgba(100,116,139,0.05); }

    /* Scrollable log container */
    .log-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 8px;
        background: #0f172a;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        font-weight: 600;
    }

    /* Dataframe improvements */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Sidebar improvements */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }
    section[data-testid="stSidebar"] .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }

    /* Connection status */
    .conn-status {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        font-size: 0.85rem;
    }
    .conn-ok { background: #064e3b; border: 1px solid #34d399; color: #34d399; }
    .conn-fail { background: #450a0a; border: 1px solid #f87171; color: #f87171; }
    .conn-wait { background: #422006; border: 1px solid #fbbf24; color: #fbbf24; }

    /* Hide streamlit branding */
    footer { display: none !important; }
    #MainMenu { visibility: hidden; }

    /* Top Movers styling */
    .mover-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 4px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .mover-symbol {
        font-weight: 700;
        font-size: 1rem;
        color: #e2e8f0;
    }
    .mover-price {
        font-size: 0.95rem;
        color: #94a3b8;
    }
    .mover-change-up {
        color: #34d399;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .mover-change-down {
        color: #f87171;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .mover-rank {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
        min-width: 24px;
    }

    /* Live Activity Feed */
    .activity-feed {
        max-height: 600px;
        overflow-y: auto;
        padding: 8px;
        background: #0a0e17;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }
    .activity-item {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        font-size: 0.82rem;
        line-height: 1.5;
        border-left: 4px solid transparent;
        animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .activity-scanner {
        background: rgba(96,165,250,0.08);
        border-left-color: #60a5fa;
    }
    .activity-analyst {
        background: rgba(167,139,250,0.08);
        border-left-color: #a78bfa;
    }
    .activity-risk {
        background: rgba(251,191,36,0.08);
        border-left-color: #fbbf24;
    }
    .activity-execution {
        background: rgba(52,211,153,0.08);
        border-left-color: #34d399;
    }
    .activity-portfolio {
        background: rgba(244,114,182,0.08);
        border-left-color: #f472b6;
    }
    .activity-tool {
        background: rgba(100,116,139,0.05);
        border-left-color: #475569;
    }
    .activity-error {
        background: rgba(248,113,113,0.1);
        border-left-color: #f87171;
    }
    .activity-success {
        background: rgba(52,211,153,0.1);
        border-left-color: #34d399;
    }

    /* Pipeline progress */
    .pipeline-step {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px;
    }
    .step-active {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: white;
        box-shadow: 0 0 12px rgba(59,130,246,0.4);
    }
    .step-done {
        background: #064e3b;
        color: #34d399;
    }
    .step-pending {
        background: #1e293b;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the main header with status bar using Streamlit columns"""
    st.markdown('<p class="main-title">Alpha-Prime</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Autonomous Intraday Trading | NSE India</p>',
               unsafe_allow_html=True)

    now = datetime.now(IST)
    is_open = is_market_open()
    is_pre = is_pre_market()

    # Status row using columns for reliable rendering
    s1, s2, s3, s4, s5 = st.columns(5)

    with s1:
        if is_open:
            st.markdown('<span class="badge badge-green">MARKET OPEN</span>',
                       unsafe_allow_html=True)
        elif is_pre:
            st.markdown('<span class="badge badge-yellow">PRE-MARKET</span>',
                       unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-red">MARKET CLOSED</span>',
                       unsafe_allow_html=True)

    with s2:
        st.markdown(f'<span style="color:#e2e8f0;font-weight:600;font-size:0.95rem;">'
                   f'{now.strftime("%H:%M:%S")} IST</span>', unsafe_allow_html=True)

    with s3:
        mode = settings.trading.mode.upper()
        if mode == "LIVE":
            st.markdown('<span class="badge badge-red">LIVE TRADING</span>',
                       unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-yellow">PAPER TRADING</span>',
                       unsafe_allow_html=True)

    with s4:
        if trading_scheduler.is_running():
            st.markdown('<span class="badge badge-green">SCHEDULER ON</span>',
                       unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-blue">SCHEDULER OFF</span>',
                       unsafe_allow_html=True)

    with s5:
        groww_status = get_system_state("groww_auth_status", "not connected")
        if "authenticated" in groww_status:
            st.markdown('<span class="badge badge-green">GROWW OK</span>',
                       unsafe_allow_html=True)
        elif "failed" in groww_status:
            st.markdown('<span class="badge badge-red">GROWW ERROR</span>',
                       unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-blue">GROWW N/A</span>',
                       unsafe_allow_html=True)

    st.markdown("---")


def render_sidebar():
    """Sidebar with controls"""
    with st.sidebar:
        st.markdown("### Control Panel")

        # ---- Groww API Keys ----
        st.markdown("---")
        st.markdown("#### Groww API Keys")
        st.caption("Keys expire daily. Update here when renewed.")

        new_api_key = st.text_input("API Key", value=settings.groww.api_key,
                                    type="password", key="groww_api_input")
        new_secret = st.text_input("Secret Key", value=settings.groww.secret_key,
                                   type="password", key="groww_secret_input")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Update Keys", width="stretch", type="primary"):
                try:
                    settings.update_groww_keys(new_api_key, new_secret)
                    reset_broker()
                    st.success("Keys saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")
        with col2:
            if st.button("Test Connection", width="stretch"):
                with st.spinner("Connecting to Groww..."):
                    try:
                        # Save keys first, then test
                        settings.update_groww_keys(new_api_key, new_secret)
                        reset_broker()
                        broker = GrowwBroker()
                        broker._ensure_client()
                        funds = broker.get_funds()
                        if funds and "error" not in funds:
                            st.success("Connected to Groww!")
                            st.json(funds)
                        else:
                            profile = broker.get_user_profile()
                            if profile and "error" not in profile:
                                st.success("Connected!")
                                st.json(profile)
                            else:
                                st.error(f"Auth OK but API error: {funds.get('error', 'Unknown')}")
                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")

        # Groww connection status
        groww_auth = get_system_state("groww_auth_status", "not connected")
        groww_time = get_system_state("groww_auth_time", "")
        if "authenticated" in groww_auth:
            st.markdown(f'<div class="conn-status conn-ok">Connected{" at " + groww_time[-8:] if groww_time else ""}</div>',
                       unsafe_allow_html=True)
        elif "failed" in groww_auth:
            st.markdown(f'<div class="conn-status conn-fail">{groww_auth}</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="conn-status conn-wait">Not connected</div>',
                       unsafe_allow_html=True)

        # ---- OpenAI Settings ----
        st.markdown("---")
        st.markdown("#### OpenAI Settings")
        openai_key = st.text_input("API Key", value=settings.openai.api_key,
                                   type="password", key="openai_key_input")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                           index=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"].index(settings.openai.model)
                           if settings.openai.model in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"] else 0)
        if st.button("Save OpenAI Settings", width="stretch"):
            try:
                settings.update_openai_settings(openai_key, model)
                st.success("OpenAI settings saved!")
            except Exception as e:
                st.error(f"Save failed: {e}")

        # ---- Trading Mode ----
        st.markdown("---")
        st.markdown("#### Trading Mode")
        mode = st.radio("Select Mode", ["Paper", "Live"],
                       index=0 if settings.trading.mode == "paper" else 1,
                       horizontal=True)
        new_mode = "paper" if mode == "Paper" else "live"
        if new_mode != settings.trading.mode:
            settings.trading.mode = new_mode
            reset_broker()  # ensure next get_broker() uses the selected mode
        else:
            settings.trading.mode = new_mode

        # ---- Scheduler ----
        st.markdown("---")
        st.markdown("#### Scheduler")

        col1, col2 = st.columns(2)
        with col1:
            start_disabled = trading_scheduler.is_running()
            if st.button("Start" if not start_disabled else "Running",
                        disabled=start_disabled, width="stretch", type="primary"):
                trading_scheduler.start()
                st.rerun()
        with col2:
            if st.button("Stop", disabled=not trading_scheduler.is_running(),
                        width="stretch"):
                trading_scheduler.stop()
                st.rerun()

        if st.button("Run Pipeline NOW", width="stretch",
                    type="secondary", help="Manually trigger one trading cycle"):
            with st.spinner("Running AI trading pipeline..."):
                trading_scheduler.run_now()
            st.success("Pipeline triggered!")

        if st.button("Square Off ALL Positions", width="stretch",
                    help="Sell all intraday positions immediately"):
            with st.spinner("Squaring off all positions..."):
                trading_scheduler._auto_square_off()
            st.success("Square-off executed! Check Agent Logs for details.")
            st.rerun()

        interval = st.slider("Interval (min)", 5, 60,
                            settings.market.analysis_interval_minutes, 5)
        settings.market.analysis_interval_minutes = interval

        # ---- Risk Settings ----
        st.markdown("---")
        st.markdown("#### Risk Parameters")
        settings.trading.max_daily_loss_pct = st.number_input(
            "Max Daily Loss %", 0.5, 10.0, settings.trading.max_daily_loss_pct, 0.5)
        settings.trading.max_position_size_pct = st.number_input(
            "Max Position Size %", 2.0, 50.0, settings.trading.max_position_size_pct, 1.0)
        settings.trading.max_open_positions = st.number_input(
            "Max Open Positions", 1, 20, settings.trading.max_open_positions, 1)
        settings.trading.capital = st.number_input(
            "Trading Capital (Rs.)", 1000.0, 10000000.0, settings.trading.capital, 1000.0)


def render_portfolio_tab():
    """Portfolio Overview"""

    # Try to get portfolio data
    try:
        broker = get_broker()
        summary = broker.get_portfolio_summary()
        error_msg = summary.get("error", "")
    except Exception as e:
        error_msg = str(e)
        summary = {
            "total_portfolio_value": 0, "total_pnl": 0,
            "total_pnl_pct": 0, "available_cash": 0,
            "total_invested": 0, "num_positions": 0,
            "positions": {}, "mode": settings.trading.mode.upper(),
            "capital": settings.trading.capital
        }

    if error_msg:
        st.warning(f"Could not fetch live data: {error_msg}")

    # Key Metrics - use compact formatting to prevent truncation
    def fmt_rs(val):
        """Format rupee amount compactly"""
        if abs(val) >= 10000000:  # 1 crore+
            return f"{val/10000000:.2f} Cr"
        elif abs(val) >= 100000:  # 1 lakh+
            return f"{val/100000:.2f} L"
        elif abs(val) >= 1000:
            return f"{val/1000:.1f}K"
        else:
            return f"{val:,.0f}"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        val = summary.get('total_portfolio_value', summary.get('capital', 0))
        st.metric("Portfolio Value", f"Rs {fmt_rs(val)}")
    with col2:
        pnl = summary.get('total_pnl', 0)
        pnl_pct = summary.get('total_pnl_pct', 0)
        st.metric("Total P&L", f"Rs {fmt_rs(pnl)}", delta=f"{pnl_pct:+.2f}%")
    with col3:
        st.metric("Available Cash", f"Rs {fmt_rs(summary.get('available_cash', 0))}")
    with col4:
        st.metric("Invested", f"Rs {fmt_rs(summary.get('total_invested', 0))}")
    with col5:
        st.metric("Positions", summary.get('num_positions', 0))

    st.markdown("")  # spacer

    col_chart, col_pos = st.columns([3, 2])

    with col_chart:
        st.markdown("##### Portfolio Value History")
        history = get_portfolio_history(limit=200)
        if history:
            hist_df = pd.DataFrame(history)
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df["timestamp"], y=hist_df["total_value"],
                mode="lines",
                name="Value",
                line=dict(color="#60a5fa", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(96, 165, 250, 0.08)"
            ))
            fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True,
                          title="Rs.", tickformat=","),
                font=dict(color="#94a3b8"),
                showlegend=False
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No portfolio history yet. Run a trading cycle to see data.")

    with col_pos:
        st.markdown("##### Open Positions (Intraday)")
        positions = summary.get("positions", {})
        if positions:
            pos_rows = []
            for sym, d in positions.items():
                pnl_val = d.get("pnl", 0)
                pos_rows.append({
                    "Symbol": sym,
                    "Qty": d.get("quantity", 0),
                    "Avg": f"{d.get('avg_price', 0):.2f}",
                    "CMP": f"{d.get('current_price', 0):.2f}",
                    "P&L": f"{pnl_val:+.2f}",
                    "P&L%": f"{d.get('pnl_pct', 0):+.1f}%"
                })
            st.dataframe(pd.DataFrame(pos_rows), width="stretch",
                        hide_index=True, height=300)
        else:
            st.info("No intraday positions")

    # Holdings section (delivery)
    holdings = summary.get("holdings", [])
    if holdings:
        st.markdown("##### Holdings (Delivery)")
        h_rows = []
        for h in holdings:
            pnl_val = h.get("pnl", 0)
            h_rows.append({
                "Symbol": h.get("symbol", ""),
                "Qty": h.get("quantity", 0),
                "Avg Price": f"Rs {h.get('avg_price', 0):.2f}",
                "CMP": f"Rs {h.get('current_price', 0):.2f}",
                "Invested": f"Rs {h.get('invested', 0):,.0f}",
                "Current": f"Rs {h.get('current_value', 0):,.0f}",
                "P&L": f"Rs {pnl_val:+,.0f}",
                "P&L%": f"{h.get('pnl_pct', 0):+.1f}%"
            })
        st.dataframe(pd.DataFrame(h_rows), width="stretch",
                    hide_index=True, height=250)

    # Position Monitor Status
    monitor_status = position_monitor.get_status()
    if monitor_status:
        st.markdown("##### Position Monitor (Trailing Stop-Loss)")
        mon_rows = []
        for sym, data in monitor_status.items():
            current_price = market_data.get_live_price(sym) or 0
            pnl = current_price - data["entry"] if current_price > 0 else 0
            pnl_pct = (pnl / data["entry"] * 100) if data["entry"] > 0 else 0
            mon_rows.append({
                "Symbol": sym,
                "Entry": f"{data['entry']:.2f}",
                "CMP": f"{current_price:.2f}" if current_price > 0 else "N/A",
                "Peak": f"{data['peak']:.2f}",
                "Trail SL": f"{data['stop_loss']:.2f}",
                "Target": f"{data['target']:.2f}",
                "P&L": f"{pnl:+.2f}",
                "P&L%": f"{pnl_pct:+.1f}%",
            })
        st.dataframe(pd.DataFrame(mon_rows), width="stretch",
                    hide_index=True, height=200)
        st.caption("Positions are auto-sold when price drops below Trail SL or reaches Target. Auto square-off at 3:15 PM.")

    # ---- Risk-Adjusted Performance Metrics (Sharpe Ratio & Max Drawdown) ----
    st.markdown("##### Risk-Adjusted Performance")
    try:
        trades_for_metrics = get_trades(limit=500, trading_mode=settings.trading.mode)
        if trades_for_metrics and len(trades_for_metrics) >= 2:
            # Calculate per-trade returns
            trade_returns = []
            for t in trades_for_metrics:
                if t.get("action") == "SELL" and t.get("pnl") is not None:
                    entry_val = t.get("price", 0) * t.get("quantity", 0)
                    if entry_val > 0:
                        ret = t["pnl"] / entry_val
                        trade_returns.append(ret)

            # Also compute from portfolio history for equity curve
            history = get_portfolio_history(limit=500)

            if trade_returns and len(trade_returns) >= 2:
                import numpy as np_metrics
                returns_arr = np_metrics.array(trade_returns)

                # Sharpe Ratio (annualized, assuming ~250 trading days, ~5 trades/day avg)
                avg_return = np_metrics.mean(returns_arr)
                std_return = np_metrics.std(returns_arr, ddof=1)
                # Risk-free rate: ~7% annually for India (RBI repo rate) -> per trade ~0.028%
                risk_free_per_trade = 0.07 / (250 * 5)  # rough approximation
                sharpe = ((avg_return - risk_free_per_trade) / std_return * np_metrics.sqrt(250 * 5)
                         if std_return > 0 else 0)

                # Win Rate
                wins = sum(1 for r in returns_arr if r > 0)
                win_rate = wins / len(returns_arr) * 100

                # Average Win / Average Loss
                win_returns = [r for r in returns_arr if r > 0]
                loss_returns = [r for r in returns_arr if r < 0]
                avg_win = np_metrics.mean(win_returns) * 100 if win_returns else 0
                avg_loss = np_metrics.mean(loss_returns) * 100 if loss_returns else 0

                # Profit Factor
                gross_profit = sum(r for r in returns_arr if r > 0)
                gross_loss = abs(sum(r for r in returns_arr if r < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

                # Max Drawdown from equity curve (portfolio history)
                max_drawdown_pct = 0.0
                if history and len(history) >= 2:
                    values = [h.get("total_value", 0) for h in reversed(history) if h.get("total_value", 0) > 0]
                    if values:
                        peak = values[0]
                        for v in values:
                            if v > peak:
                                peak = v
                            dd = (peak - v) / peak * 100 if peak > 0 else 0
                            if dd > max_drawdown_pct:
                                max_drawdown_pct = dd

                # Display metrics
                mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                with mc1:
                    sharpe_color = "normal" if sharpe >= 1 else "off"
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}",
                             help="Risk-adjusted return. >1 = good, >2 = excellent")
                with mc2:
                    st.metric("Win Rate", f"{win_rate:.0f}%",
                             help="Percentage of profitable trades")
                with mc3:
                    st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor < 100 else "INF",
                             help="Gross profit / Gross loss. >1.5 = good")
                with mc4:
                    st.metric("Avg Win", f"+{avg_win:.2f}%",
                             help="Average return on winning trades")
                with mc5:
                    st.metric("Avg Loss", f"{avg_loss:.2f}%",
                             help="Average return on losing trades")
                with mc6:
                    st.metric("Max Drawdown", f"-{max_drawdown_pct:.2f}%",
                             help="Largest peak-to-trough decline")

                # Equity curve with drawdown overlay
                if history and len(history) >= 3:
                    hist_df2 = pd.DataFrame(reversed(history))
                    hist_df2["timestamp"] = pd.to_datetime(hist_df2["timestamp"])
                    values_series = hist_df2["total_value"]
                    peak_series = values_series.cummax()
                    drawdown_series = ((peak_series - values_series) / peak_series * 100).fillna(0)

                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(
                        x=hist_df2["timestamp"], y=drawdown_series,
                        mode="lines", name="Drawdown %",
                        line=dict(color="#f87171", width=1.5),
                        fill="tozeroy", fillcolor="rgba(248, 113, 113, 0.15)"
                    ))
                    fig_dd.update_layout(
                        height=200,
                        margin=dict(l=10, r=10, t=5, b=25),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True,
                                  title="Drawdown %", autorange="reversed"),
                        font=dict(color="#94a3b8", size=10),
                        showlegend=False
                    )
                    st.plotly_chart(fig_dd, width="stretch")
            else:
                st.info("Not enough completed trades to calculate risk metrics. Complete at least 2 sell trades.")
        else:
            st.info("No trade history available. Run trading cycles to see risk-adjusted performance metrics.")
    except Exception as e:
        st.warning(f"Could not compute risk metrics: {e}")

    st.markdown("")  # spacer

    # Show Groww account details if live mode
    if settings.trading.mode == "live":
        try:
            broker = get_broker()
            if isinstance(broker, GrowwBroker):
                with st.expander("Groww Account Details", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Funds / Margin**")
                        funds = broker.get_funds()
                        if funds and "error" not in funds:
                            st.json(funds)
                        else:
                            st.warning(f"Could not load: {funds.get('error', 'Unknown')}")
                    with col2:
                        st.markdown("**User Profile**")
                        profile = broker.get_user_profile()
                        if profile and "error" not in profile:
                            st.json(profile)
                        else:
                            st.warning(f"Could not load: {profile.get('error', 'Unknown')}")
        except Exception as e:
            st.warning(f"Groww data unavailable: {e}")


def render_trades_tab():
    """Trade History (filtered by current mode: paper or live)"""
    trades = get_trades(limit=200, trading_mode=settings.trading.mode)
    if trades:
        df = pd.DataFrame(trades)
        cols = ["timestamp", "symbol", "action", "quantity", "price",
               "order_type", "status", "rationale", "agent_name", "order_id"]
        available = [c for c in cols if c in df.columns]
        st.dataframe(df[available], width="stretch", hide_index=True, height=500)
    else:
        st.info("No trades executed yet. Start the scheduler or run a manual cycle.")


def render_signals_tab():
    """Trading Signals"""
    signals = get_signals(limit=200)
    if signals:
        df = pd.DataFrame(signals)
        cols = ["timestamp", "symbol", "signal_type", "strength", "agent_name", "reasoning"]
        available = [c for c in cols if c in df.columns]

        # Color code signal types
        st.dataframe(df[available], width="stretch", hide_index=True, height=500)
    else:
        st.info("No signals generated yet.")


def render_analysis_tab():
    """Manual Stock Analysis with Charts"""
    col_input, col_chart = st.columns([1, 2])

    with col_input:
        st.markdown("##### Analyze a Stock")
        symbol = st.text_input("Stock Symbol", value="RELIANCE",
                              placeholder="e.g., RELIANCE, TCS, INFY").upper().strip()
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)
        analyze_btn = st.button("Run Analysis", type="primary", width="stretch")

    if analyze_btn and symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            df = market_data.get_historical_data(symbol, period=period)

            if df is not None and not df.empty:
                report = technical_analyzer.analyze(df, symbol=clean_symbol(symbol))

                if report:
                    with col_chart:
                        # Signal badge
                        sig_colors = {"BUY": "green", "SELL": "red", "HOLD": "orange"}
                        sig_color = sig_colors.get(report.overall_signal, "gray")
                        st.markdown(
                            f"### {symbol} - :{sig_color}[{report.overall_signal}] "
                            f"| Strength: {report.overall_strength:.0%} "
                            f"| Trend: {report.trend} "
                            f"| Price: Rs.{report.current_price:.2f}"
                        )

                        # Candlestick chart
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=df.index, open=df["Open"], high=df["High"],
                            low=df["Low"], close=df["Close"], name=symbol,
                            increasing_line_color="#34d399",
                            decreasing_line_color="#f87171"
                        ))

                        # Support / Resistance lines
                        for lvl in report.support_levels[:2]:
                            fig.add_hline(y=lvl, line_dash="dot", line_color="#34d399",
                                        opacity=0.6, annotation_text=f"S: {lvl:.0f}",
                                        annotation_position="bottom right")
                        for lvl in report.resistance_levels[:2]:
                            fig.add_hline(y=lvl, line_dash="dot", line_color="#f87171",
                                        opacity=0.6, annotation_text=f"R: {lvl:.0f}",
                                        annotation_position="top right")

                        fig.update_layout(
                            height=480,
                            xaxis_rangeslider_visible=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                            yaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                                      title="Price (Rs.)"),
                            font=dict(color="#94a3b8"),
                            showlegend=False,
                            margin=dict(l=10, r=10, t=10, b=30),
                        )
                        st.plotly_chart(fig, width="stretch")

                    # Indicators table below
                    st.markdown("---")
                    col_ind, col_pat, col_lvl = st.columns(3)

                    with col_ind:
                        st.markdown("##### Technical Indicators")
                        for sig in report.signals:
                            icons = {"BUY": "\U0001f7e2", "SELL": "\U0001f534", "NEUTRAL": "\u26aa"}
                            icon = icons.get(sig.signal, "\u26aa")
                            st.markdown(
                                f"{icon} **{sig.indicator}** - {sig.description}"
                            )

                    with col_pat:
                        st.markdown("##### Candlestick Patterns")
                        if report.patterns:
                            for pat in report.patterns:
                                icon = "\U0001f7e2" if pat.signal == "BULLISH" else "\U0001f534"
                                st.markdown(f"{icon} **{pat.pattern_name}** ({pat.strength:.0%})")
                                st.caption(pat.description)
                        else:
                            st.info("No notable patterns detected")

                    with col_lvl:
                        st.markdown("##### Key Levels")
                        if report.support_levels:
                            st.markdown("**Support:**")
                            for s in report.support_levels[:3]:
                                st.markdown(f"  \U0001f7e2 Rs. {s:,.2f}")
                        if report.resistance_levels:
                            st.markdown("**Resistance:**")
                            for r in report.resistance_levels[:3]:
                                st.markdown(f"  \U0001f534 Rs. {r:,.2f}")
                else:
                    with col_chart:
                        st.error("Analysis failed - insufficient data")
            else:
                with col_chart:
                    st.error(f"No data found for {symbol}. Check the symbol name.")


def render_agent_logs_tab():
    """Agent Activity Logs with color-coded entries"""
    col_filter, col_logs = st.columns([1, 4])

    with col_filter:
        agent_filter = st.selectbox("Agent", [
            "All", "market_scanner", "technical_analyst", "risk_manager",
            "execution_agent", "portfolio_manager", "scheduler", "broker", "pipeline"
        ])
        log_limit = st.slider("Show last", 20, 500, 100, 20)
        if st.button("Refresh Logs", width="stretch"):
            st.rerun()

    with col_logs:
        agent_name = None if agent_filter == "All" else agent_filter
        logs = get_agent_logs(agent_name=agent_name, limit=log_limit)

        if logs:
            log_class_map = {
                "market_scanner": "log-scanner",
                "technical_analyst": "log-analyst",
                "risk_manager": "log-risk",
                "execution_agent": "log-execution",
                "error": "log-error",
                "trade": "log-trade",
            }

            html = '<div class="log-container">'
            for log in logs:
                agent = log.get("agent_name", "")
                log_type = log.get("log_type", "")
                ts = log.get("timestamp", "")[-8:]  # just time
                msg = log.get("message", "")

                css_class = log_class_map.get(agent, log_class_map.get(log_type, "log-default"))
                if log_type == "error":
                    css_class = "log-error"

                html += (
                    f'<div class="log-entry {css_class}">'
                    f'<span style="color:#64748b">{ts}</span> '
                    f'<span style="font-weight:700">[{agent}]</span> '
                    f'<span style="color:#94a3b8">{log_type}</span> '
                    f'{msg}'
                    f'</div>'
                )
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No agent logs yet. Run a trading cycle to see agent activity here.")


def render_watchlist_tab():
    """Watchlist Management"""
    col_add, col_list = st.columns([1, 3])

    with col_add:
        st.markdown("##### Add to Watchlist")
        new_sym = st.text_input("Symbol", placeholder="RELIANCE").upper().strip()
        new_sector = st.text_input("Sector", placeholder="Oil & Gas")
        new_notes = st.text_input("Notes", placeholder="Reason")
        if st.button("Add", width="stretch", type="primary"):
            if new_sym:
                add_to_watchlist(clean_symbol(new_sym), new_sector, new_notes)
                st.success(f"Added {new_sym}")
                st.rerun()

        st.markdown("---")
        if st.button("Add Top NIFTY 50", width="stretch"):
            for sym in market_data.get_nifty50_stocks()[:10]:
                add_to_watchlist(sym, "NIFTY 50", "Auto-added")
            st.success("Added top 10 NIFTY 50 stocks")
            st.rerun()

    with col_list:
        st.markdown("##### Watchlist")
        watchlist = get_watchlist()
        if watchlist:
            symbols = [w["symbol"] for w in watchlist]
            prices = market_data.get_multiple_prices(symbols)

            rows = []
            for w in watchlist:
                sym = w["symbol"]
                price = prices.get(sym, None)
                rows.append({
                    "Symbol": sym,
                    "Price": f"Rs. {price:,.2f}" if price else "N/A",
                    "Sector": w.get("sector", ""),
                    "Notes": w.get("notes", ""),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch",
                        hide_index=True, height=400)

            remove_sym = st.selectbox("Remove symbol", ["(select)"] + symbols)
            if remove_sym != "(select)" and st.button("Remove Selected"):
                remove_from_watchlist(remove_sym)
                st.rerun()
        else:
            st.info("Watchlist empty. Add stocks above.")


def render_top_movers_tab():
    """Top Movers - shows top gainers/losers across multiple indices with Run Pipeline button"""

    # Index selector and view mode
    col_index, col_view, _ = st.columns([2, 1, 2])
    with col_index:
        selected_index = st.selectbox(
            "Select Index",
            ["NIFTY 50", "NIFTY 100", "NIFTY 500", "NIFTY Midcap 100",
             "NIFTY Smallcap 100", "Nifty Total Market"],
            key="index_selector"
        )
    with col_view:
        view_mode = st.radio("View", ["Gainers", "Losers"], horizontal=True, key="movers_view")

    st.markdown(f"##### Top Movers Today ({selected_index})")

    # Fetch data for selected index stocks
    index_stocks = market_data.get_index_stocks(selected_index)
    stock_count = len(index_stocks)

    with st.spinner(f"Fetching live prices for {stock_count} {selected_index} stocks..."):
        movers_data = []

        # Use yfinance to get OHLC data (open + current price for % change)
        import yfinance as yf

        # For large indices, fetch in batches to avoid yfinance overload
        batch_size = 50
        for batch_start in range(0, len(index_stocks), batch_size):
            batch = index_stocks[batch_start:batch_start + batch_size]
            symbols_ns = [f"{s}.NS" for s in batch]

            try:
                tickers = yf.Tickers(" ".join(symbols_ns))
                for sym, sym_ns in zip(batch, symbols_ns):
                    try:
                        info = tickers.tickers[sym_ns].fast_info
                        last_price = getattr(info, 'last_price', 0) or 0
                        open_price = getattr(info, 'open', 0) or 0
                        prev_close = getattr(info, 'previous_close', 0) or 0

                        if prev_close > 0 and last_price > 0:
                            change = last_price - prev_close
                            change_pct = (change / prev_close) * 100
                            movers_data.append({
                                "symbol": sym,
                                "price": last_price,
                                "change": change,
                                "change_pct": change_pct,
                                "open": open_price,
                                "prev_close": prev_close,
                                "volume": getattr(info, 'last_volume', 0) or 0
                            })
                    except Exception:
                        pass
            except Exception as e:
                st.warning(f"Error fetching batch starting at {batch_start}: {e}")

    if not movers_data:
        st.warning(f"Could not fetch market data for {selected_index}. Market may be closed or data unavailable.")
        return

    st.caption(f"Showing top 10 of {len(movers_data)} stocks fetched from {stock_count} in {selected_index}")

    # Sort by change_pct
    if view_mode == "Gainers":
        movers_data.sort(key=lambda x: x["change_pct"], reverse=True)
    else:
        movers_data.sort(key=lambda x: x["change_pct"])

    top_10 = movers_data[:10]

    # Display as a table with nice formatting
    col_table, col_action = st.columns([3, 1])

    with col_table:
        rows = []
        for i, m in enumerate(top_10):
            change_str = f"+{m['change']:.2f}" if m['change'] >= 0 else f"{m['change']:.2f}"
            pct_str = f"+{m['change_pct']:.2f}%" if m['change_pct'] >= 0 else f"{m['change_pct']:.2f}%"
            vol_str = f"{m['volume']:,.0f}" if m['volume'] else "N/A"
            rows.append({
                "#": i + 1,
                "Symbol": m["symbol"],
                "Price": f"{m['price']:.2f}",
                "Change": change_str,
                "Change %": pct_str,
                "Open": f"{m['open']:.2f}",
                "Prev Close": f"{m['prev_close']:.2f}",
                "Volume": vol_str,
            })

        df = pd.DataFrame(rows)

        # Color the dataframe
        def color_change(val):
            if isinstance(val, str):
                if val.startswith("+"):
                    return "color: #34d399; font-weight: 700"
                elif val.startswith("-"):
                    return "color: #f87171; font-weight: 700"
            return ""

        styled = df.style.map(color_change, subset=["Change", "Change %"])
        st.dataframe(styled, width="stretch", hide_index=True, height=420)

    with col_action:
        st.markdown("##### Quick Actions")
        st.markdown("")

        # Store top movers in session state for pipeline use
        st.session_state["top_movers"] = top_10

        selected_stocks = st.multiselect(
            "Select stocks to trade",
            [m["symbol"] for m in top_10],
            default=[m["symbol"] for m in top_10[:3]],
            key="movers_select"
        )

        st.markdown("")

        if st.button("Run Pipeline on Selected", type="primary", width="stretch",
                     help="Run AI trading pipeline focused on selected top movers"):
            if selected_stocks:
                st.session_state["pipeline_running"] = True
                st.session_state["pipeline_stocks"] = selected_stocks
                _run_pipeline_on_movers(selected_stocks)
            else:
                st.warning("Select at least one stock")

        st.markdown("")
        st.markdown("---")
        st.markdown("")

        def _trigger_analyze():
            sel = st.session_state.get("movers_select", [])
            if sel:
                st.session_state["run_analysis_stocks"] = list(sel)
            else:
                st.session_state["analysis_no_selection"] = True

        if st.button("Analyze Selected", width="stretch",
                     help="Run technical analysis on selected stocks",
                     key="btn_analyze_selected",
                     on_click=_trigger_analyze):
            pass  # Callback handles it

        if st.session_state.pop("analysis_no_selection", False):
            st.warning("Select at least one stock")

    # Run analysis if triggered (outside columns so spinner + result render full-width)
    if "run_analysis_stocks" in st.session_state and st.session_state["run_analysis_stocks"]:
        analysis_stocks = st.session_state.pop("run_analysis_stocks")
        st.markdown("---")
        st.markdown(f"##### Analyzing: {', '.join(analysis_stocks)}")
        with st.spinner(f"Running AI technical analysis on {', '.join(analysis_stocks)}... This may take 1-2 minutes."):
            try:
                from alpha_prime.agents.pipeline import run_analysis
                result = run_analysis(analysis_stocks)
                st.session_state["analysis_result"] = result
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.session_state["analysis_result"] = None

    # Show analysis result if available
    analysis_result = st.session_state.get("analysis_result")
    if analysis_result is not None:
        st.markdown("---")
        with st.expander("Analysis Result", expanded=True):
            if analysis_result:
                st.markdown(analysis_result)
            else:
                st.info("Analysis completed but returned no content. Check OpenAI API key and try again.")
            if st.button("Clear Analysis", key="clear_analysis"):
                st.session_state["analysis_result"] = None
                st.rerun()


def _run_pipeline_on_movers(stocks: list):
    """Run the full pipeline focused on specific top mover stocks"""
    from alpha_prime.agents.pipeline import (
        get_llm, run_agent_node, create_initial_state,
        TradingState, get_scanner_prompt, get_analyst_prompt,
        get_risk_manager_prompt, get_execution_prompt
    )
    from alpha_prime.agents.tools import (
        MARKET_ANALYSIS_TOOLS, TRADING_TOOLS, SIGNAL_TOOLS
    )
    from alpha_prime.core.database import log_agent_activity

    symbols_str = ", ".join(stocks)
    progress = st.progress(0, text="Starting pipeline...")
    status_container = st.container()

    state = create_initial_state()

    # Phase 1: Scanner (but focused on selected stocks)
    with status_container:
        st.markdown(_pipeline_status_html("scanner"), unsafe_allow_html=True)
    progress.progress(10, text="Phase 1: Scanning selected stocks...")

    scanner_prompt = get_scanner_prompt()
    scanner_msg = f"""The user has selected these TOP MOVERS to trade: {symbols_str}

Execute these steps:
1. Call check_market_status
2. Call get_multiple_stock_prices with "{','.join(stocks)}"
3. Call run_intraday_analysis on each stock
4. ALL of these stocks are pre-selected by the user as top movers. Output them as BUY candidates with analysis."""

    scanner_output = run_agent_node(
        state, scanner_prompt, scanner_msg,
        MARKET_ANALYSIS_TOOLS + SIGNAL_TOOLS,
        "market_scanner", max_iterations=15
    )
    state["scanner_output"] = scanner_output
    progress.progress(30, text="Phase 1 complete. Starting analysis...")

    # Phase 2: Analyst
    with status_container:
        st.markdown(_pipeline_status_html("analyst"), unsafe_allow_html=True)
    progress.progress(35, text="Phase 2: Deep technical analysis...")

    analyst_prompt = get_analyst_prompt()
    analyst_msg = f"""The scanner identified these TOP MOVERS selected by the user:

{scanner_output}

For EACH stock:
1. Run both daily and intraday analysis
2. Compute exact entry, stop-loss, target
3. Rate signal strength
4. Save signals using save_signal tool"""

    analyst_output = run_agent_node(
        state, analyst_prompt, analyst_msg,
        MARKET_ANALYSIS_TOOLS + SIGNAL_TOOLS,
        "technical_analyst", max_iterations=15
    )
    state["analyst_output"] = analyst_output
    progress.progress(55, text="Phase 2 complete. Risk check...")

    # Phase 3: Risk Manager
    with status_container:
        st.markdown(_pipeline_status_html("risk"), unsafe_allow_html=True)
    progress.progress(60, text="Phase 3: Risk validation & position sizing...")

    risk_prompt = get_risk_manager_prompt()
    risk_msg = f"""Review these trade proposals from analysis:

{analyst_output}

Check risk rules and APPROVE each valid trade with position size."""

    risk_tools = MARKET_ANALYSIS_TOOLS + [
        t for t in TRADING_TOOLS if t.name != "place_trade"
    ] + SIGNAL_TOOLS
    risk_output = run_agent_node(
        state, risk_prompt, risk_msg,
        risk_tools, "risk_manager", max_iterations=10
    )
    state["risk_output"] = risk_output
    progress.progress(80, text="Phase 3 complete. Executing trades...")

    # Phase 4: Executor
    with status_container:
        st.markdown(_pipeline_status_html("execution"), unsafe_allow_html=True)
    progress.progress(85, text="Phase 4: Placing orders via Groww...")

    from alpha_prime.agents.prompts import get_execution_prompt
    exec_prompt = get_execution_prompt()
    exec_msg = f"""Risk Manager approved trades:

{risk_output}

EXECUTE every APPROVED trade using place_trade. Do NOT skip any."""

    exec_output = run_agent_node(
        state, exec_prompt, exec_msg,
        TRADING_TOOLS + MARKET_ANALYSIS_TOOLS,
        "execution_agent", max_iterations=10
    )
    state["execution_output"] = exec_output
    progress.progress(100, text="Pipeline complete!")

    st.session_state["pipeline_running"] = False

    # Show results
    st.success("Pipeline completed!")
    with st.expander("Scanner Output", expanded=False):
        st.markdown(scanner_output)
    with st.expander("Analyst Output", expanded=False):
        st.markdown(analyst_output)
    with st.expander("Risk Manager Output", expanded=False):
        st.markdown(risk_output)
    with st.expander("Execution Output", expanded=True):
        st.markdown(exec_output)


def _pipeline_status_html(active_phase: str) -> str:
    """Generate pipeline progress bar HTML"""
    phases = [
        ("scanner", "Scanner", "Scanning stocks"),
        ("analyst", "Analyst", "Technical analysis"),
        ("risk", "Risk Mgr", "Risk validation"),
        ("execution", "Executor", "Placing trades"),
    ]
    phase_order = [p[0] for p in phases]
    active_idx = phase_order.index(active_phase) if active_phase in phase_order else -1

    html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin:8px 0;">'
    for i, (key, label, desc) in enumerate(phases):
        if i < active_idx:
            css = "step-done"
            icon = "&#10003;"
        elif i == active_idx:
            css = "step-active"
            icon = "&#9654;"
        else:
            css = "step-pending"
            icon = "&#9679;"
        html += f'<span class="pipeline-step {css}">{icon} {label}</span>'

    # Arrow connectors
    html += '</div>'
    return html


def render_live_activity_tab():
    """Live Pipeline Activity - shows real-time agent-to-agent communication"""
    st.markdown("##### Live Pipeline Activity")
    st.caption("Shows real-time agent activity, tool calls, and decisions during pipeline execution")

    col_filter, col_refresh = st.columns([3, 1])
    with col_filter:
        show_tools = st.checkbox("Show tool calls", value=True, key="show_tools_activity")
        show_limit = st.slider("Show last N entries", 20, 500, 100, 10, key="activity_limit")
    with col_refresh:
        if st.button("Refresh", width="stretch", key="refresh_activity"):
            st.rerun()

    # Fetch recent agent logs
    logs = get_agent_logs(limit=show_limit)

    if not logs:
        st.info("No pipeline activity yet. Run a trading cycle to see live agent activity here.")
        return

    # Pipeline progress indicator (from latest logs)
    latest_agents = set()
    for log in logs[:50]:
        agent = log.get("agent_name", "")
        if log.get("log_type") == "complete":
            latest_agents.add(agent)

    # Render activity feed
    agent_css_map = {
        "market_scanner": ("activity-scanner", "SCANNER", "#60a5fa"),
        "technical_analyst": ("activity-analyst", "ANALYST", "#a78bfa"),
        "risk_manager": ("activity-risk", "RISK MGR", "#fbbf24"),
        "execution_agent": ("activity-execution", "EXECUTOR", "#34d399"),
        "portfolio_manager": ("activity-portfolio", "REVIEW", "#f472b6"),
        "pipeline": ("activity-success", "PIPELINE", "#34d399"),
        "scheduler": ("activity-tool", "SCHEDULER", "#64748b"),
        "broker": ("activity-tool", "BROKER", "#64748b"),
    }

    html = '<div class="activity-feed">'

    for log in logs:
        agent = log.get("agent_name", "unknown")
        log_type = log.get("log_type", "")
        message = log.get("message", "")
        ts = log.get("timestamp", "")
        details = log.get("details_json", "")

        # Skip tool calls if unchecked
        if not show_tools and log_type == "tool_call":
            continue

        css_class, agent_label, agent_color = agent_css_map.get(
            agent, ("activity-tool", agent.upper(), "#64748b")
        )

        if log_type == "error":
            css_class = "activity-error"

        # Time
        time_str = ts[-8:] if len(ts) >= 8 else ts

        # Icon based on type
        icons = {
            "start": "&#9654;",      # play
            "tool_call": "&#128295;",  # wrench
            "complete": "&#10003;",    # check
            "error": "&#10007;",       # cross
            "trade": "&#128176;",      # money
        }
        icon = icons.get(log_type, "&#9679;")

        # Parse tool details
        detail_html = ""
        if log_type == "tool_call" and details:
            try:
                d = json.loads(details) if isinstance(details, str) else details
                if isinstance(d, dict):
                    tool_args = ", ".join(f"{k}={v}" for k, v in list(d.items())[:3])
                    detail_html = f'<br><span style="color:#475569;font-size:0.75rem;font-family:monospace;">  args: {tool_args}</span>'
            except Exception:
                pass

        html += (
            f'<div class="activity-item {css_class}">'
            f'<span style="color:#475569;font-size:0.75rem;">{time_str}</span> '
            f'{icon} '
            f'<span style="color:{agent_color};font-weight:700;font-size:0.8rem;">[{agent_label}]</span> '
            f'<span style="color:#94a3b8;font-size:0.78rem;">{log_type}</span> '
            f'<span style="color:#cbd5e1;font-size:0.82rem;">{message}</span>'
            f'{detail_html}'
            f'</div>'
        )

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    # Stats summary
    st.markdown("---")
    st.markdown("##### Activity Summary")
    agent_counts = {}
    tool_counts = {}
    error_count = 0
    for log in logs:
        agent = log.get("agent_name", "unknown")
        agent_counts[agent] = agent_counts.get(agent, 0) + 1
        if log.get("log_type") == "tool_call":
            msg = log.get("message", "")
            tool_name = msg.replace("Calling ", "").split("(")[0].strip() if "Calling" in msg else msg
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        if log.get("log_type") == "error":
            error_count += 1

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Agent Activity**")
        for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
            _, label, color = agent_css_map.get(agent, ("", agent, "#64748b"))
            st.markdown(f'<span style="color:{color};font-weight:600;">{label}</span>: {count} events',
                       unsafe_allow_html=True)
    with col2:
        st.markdown("**Tool Calls**")
        for tool_name, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:8]:
            st.markdown(f"`{tool_name}`: {count} calls")
    with col3:
        st.metric("Total Events", len(logs))
        st.metric("Errors", error_count, delta_color="inverse")


def main():
    """Main app"""
    init_database()
    render_header()
    render_sidebar()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Portfolio", "Top Movers", "Live Activity",
        "Trades", "Signals", "Analysis", "Agent Logs", "Watchlist"
    ])

    with tab1:
        render_portfolio_tab()
    with tab2:
        render_top_movers_tab()
    with tab3:
        render_live_activity_tab()
    with tab4:
        render_trades_tab()
    with tab5:
        render_signals_tab()
    with tab6:
        render_analysis_tab()
    with tab7:
        render_agent_logs_tab()
    with tab8:
        render_watchlist_tab()

    # Auto-refresh every 30 seconds
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30000, key="auto_refresh")


if __name__ == "__main__":
    main()
