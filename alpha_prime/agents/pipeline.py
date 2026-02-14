"""
Multi-Agent Trading Pipeline using LangGraph
Orchestrates the flow: Scanner -> Analyst -> Risk Manager -> Executor

This is the brain of Alpha-Prime. Each agent has a specialized role,
and the pipeline ensures systematic, risk-managed trading decisions.
"""

import json
from typing import TypedDict, Annotated, Sequence, Literal
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from loguru import logger

from alpha_prime.core.config import settings
from alpha_prime.core.database import log_agent_activity
from alpha_prime.agents.tools import (
    MARKET_ANALYSIS_TOOLS, TRADING_TOOLS, SIGNAL_TOOLS, ALL_TOOLS
)
from alpha_prime.agents.prompts import (
    get_scanner_prompt, get_analyst_prompt,
    get_risk_manager_prompt, get_execution_prompt,
    get_portfolio_manager_prompt
)


# ---- State Definition ----

class TradingState(TypedDict):
    """State that flows through the trading pipeline"""
    messages: Sequence[BaseMessage]
    scanner_output: str
    analyst_output: str
    risk_output: str
    execution_output: str
    portfolio_output: str
    current_phase: str
    trade_decisions: list
    error: str


def create_initial_state() -> TradingState:
    """Create a fresh pipeline state"""
    return TradingState(
        messages=[],
        scanner_output="",
        analyst_output="",
        risk_output="",
        execution_output="",
        portfolio_output="",
        current_phase="start",
        trade_decisions=[],
        error=""
    )


# ---- Agent Factory ----

def get_llm():
    """Get the LLM instance"""
    return ChatOpenAI(
        model=settings.openai.model,
        temperature=settings.openai.temperature,
        api_key=settings.openai.api_key,
        request_timeout=120,
    )


def create_agent_with_tools(system_prompt: str, tools: list):
    """Create an agent (LLM bound with tools)"""
    llm = get_llm()
    if tools:
        return llm.bind_tools(tools), system_prompt
    return llm, system_prompt


# ---- Agent Node Functions ----

def run_agent_node(state: TradingState, system_prompt: str, user_message: str,
                   tools: list, agent_name: str, max_iterations: int = 10) -> str:
    """Run an agent node with tool calling loop"""
    llm = get_llm()

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]

    log_agent_activity(agent_name, "start", f"Agent {agent_name} starting")
    logger.info(f"[{agent_name}] Starting agent execution")

    for iteration in range(max_iterations):
        try:
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # Check for tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    logger.info(f"[{agent_name}] Calling tool: {tool_name}({json.dumps(tool_args)[:200]})")
                    log_agent_activity(agent_name, "tool_call",
                                     f"Calling {tool_name}", tool_args)

                    # Find and execute the tool
                    tool_result = "Tool not found"
                    for t in tools:
                        if t.name == tool_name:
                            try:
                                tool_result = t.invoke(tool_args)
                            except Exception as e:
                                tool_result = json.dumps({"error": str(e)})
                                logger.error(f"[{agent_name}] Tool {tool_name} error: {e}")
                            break

                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    ))
            else:
                # No more tool calls, agent is done
                final_output = response.content
                logger.info(f"[{agent_name}] Completed (iteration {iteration + 1})")
                log_agent_activity(agent_name, "complete",
                                 f"Completed after {iteration + 1} iterations",
                                 {"output_length": len(final_output)})
                return final_output

        except Exception as e:
            logger.error(f"[{agent_name}] Error in iteration {iteration}: {e}")
            log_agent_activity(agent_name, "error", str(e))
            # Fail fast on auth/config errors - no point retrying
            err_str = str(e).lower()
            if any(k in err_str for k in ["api key", "authentication", "401", "403", "api_key"]):
                return f"Agent {agent_name} failed: {str(e)}"
            if iteration == max_iterations - 1:
                return f"Agent {agent_name} encountered an error: {str(e)}"

    return "Agent reached maximum iterations"


# ---- Pipeline Node Functions ----

def scanner_node(state: TradingState) -> TradingState:
    """Market Scanner: Identifies trading opportunities"""
    logger.info("=" * 60)
    logger.info("PHASE 1: MARKET SCANNER")
    logger.info("=" * 60)

    prompt = get_scanner_prompt()
    user_msg = """Execute these steps in order:
1. Call check_market_status to verify market is open
2. Call get_multiple_stock_prices with "RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,SBIN,BHARTIARTL,ITC,KOTAKBANK,LT"
3. Call run_intraday_analysis on the top 3 stocks with highest price movement
4. Output your TOP 3 BUY candidates with reasons

You MUST identify at least 3 stocks for BUY. Do NOT output "no opportunities found"."""

    output = run_agent_node(
        state, prompt, user_msg,
        MARKET_ANALYSIS_TOOLS + SIGNAL_TOOLS,
        "market_scanner",
        max_iterations=15
    )

    state["scanner_output"] = output
    state["current_phase"] = "scanner_complete"
    return state


def analyst_node(state: TradingState) -> TradingState:
    """Technical Analyst: Deep analysis on scanner picks"""
    logger.info("=" * 60)
    logger.info("PHASE 2: TECHNICAL ANALYST")
    logger.info("=" * 60)

    prompt = get_analyst_prompt()
    user_msg = f"""The Market Scanner has identified the following opportunities:

{state['scanner_output']}

For each stock identified above:
1. Run detailed technical analysis (both daily and intraday)
2. Identify precise entry points, stop-loss, and target levels
3. Rate the signal strength
4. Provide your BUY/SELL/HOLD recommendation
5. Save signals using the save_signal tool

Be specific with price levels and use actual market data."""

    output = run_agent_node(
        state, prompt, user_msg,
        MARKET_ANALYSIS_TOOLS + SIGNAL_TOOLS,
        "technical_analyst",
        max_iterations=15
    )

    state["analyst_output"] = output
    state["current_phase"] = "analyst_complete"
    return state


def risk_node(state: TradingState) -> TradingState:
    """Risk Manager: Validates and sizes trades"""
    logger.info("=" * 60)
    logger.info("PHASE 3: RISK MANAGER")
    logger.info("=" * 60)

    prompt = get_risk_manager_prompt()
    user_msg = f"""Review the following trade proposals from the Technical Analyst:

{state['analyst_output']}

Current Portfolio Status:
Use the get_portfolio and get_current_positions tools to check the current state.

For each proposed trade:
1. Check if it complies with all risk rules
2. Calculate proper position size
3. Verify risk-reward ratio
4. APPROVE or REJECT with clear reasoning
5. If approved, specify exact: symbol, action, quantity, entry, stop-loss, target

Output a clear list of APPROVED trades with exact parameters, or explain why trades were rejected."""

    # Risk manager gets portfolio tools but NOT the place_trade tool
    risk_tools = MARKET_ANALYSIS_TOOLS + [
        t for t in TRADING_TOOLS if t.name != "place_trade"
    ] + SIGNAL_TOOLS

    output = run_agent_node(
        state, prompt, user_msg,
        risk_tools,
        "risk_manager",
        max_iterations=10
    )

    state["risk_output"] = output
    state["current_phase"] = "risk_complete"
    return state


def execution_node(state: TradingState) -> TradingState:
    """Execution Agent: Places approved trades"""
    logger.info("=" * 60)
    logger.info("PHASE 4: EXECUTION AGENT")
    logger.info("=" * 60)

    prompt = get_execution_prompt()
    user_msg = f"""The Risk Manager has approved the following trades:

{state['risk_output']}

EXECUTE every trade marked as APPROVED:
1. For each APPROVED trade, call get_stock_price to get current price
2. Call place_trade with: symbol, action="BUY", quantity, rationale="AI intraday signal", stop_loss, target
3. Report each execution result

You MUST call place_trade for each approved trade. Do not skip any.
If the risk manager output contains stock symbols with BUY and quantities, execute them ALL."""

    output = run_agent_node(
        state, prompt, user_msg,
        TRADING_TOOLS + MARKET_ANALYSIS_TOOLS,
        "execution_agent",
        max_iterations=10
    )

    state["execution_output"] = output
    state["current_phase"] = "execution_complete"
    return state


def portfolio_review_node(state: TradingState) -> TradingState:
    """Portfolio Manager: Reviews and summarizes"""
    logger.info("=" * 60)
    logger.info("PHASE 5: PORTFOLIO REVIEW")
    logger.info("=" * 60)

    prompt = get_portfolio_manager_prompt()
    user_msg = f"""Review the completed trading session:

SCANNER OUTPUT:
{state['scanner_output'][:500]}

ANALYST OUTPUT:
{state['analyst_output'][:500]}

RISK DECISIONS:
{state['risk_output'][:500]}

EXECUTION RESULTS:
{state['execution_output'][:500]}

1. Get the current portfolio status
2. Review all positions and their P&L
3. Provide a comprehensive summary
4. Make recommendations for the next session
"""

    review_tools = [t for t in TRADING_TOOLS if t.name != "place_trade"] + MARKET_ANALYSIS_TOOLS

    output = run_agent_node(
        state, prompt, user_msg,
        review_tools,
        "portfolio_manager",
        max_iterations=8
    )

    state["portfolio_output"] = output
    state["current_phase"] = "complete"
    return state


# ---- Pipeline Router ----

def should_continue(state: TradingState) -> str:
    """Route to the next node based on current phase"""
    phase = state.get("current_phase", "start")

    if phase == "start":
        return "scanner"
    elif phase == "scanner_complete":
        return "analyst"
    elif phase == "analyst_complete":
        return "risk"
    elif phase == "risk_complete":
        return "execution"
    elif phase == "execution_complete":
        return "portfolio_review"
    else:
        return END


# ---- Build the Pipeline Graph ----

def build_trading_pipeline() -> StateGraph:
    """Build and compile the multi-agent trading pipeline"""

    workflow = StateGraph(TradingState)

    # Add nodes
    workflow.add_node("scanner", scanner_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("risk", risk_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("portfolio_review", portfolio_review_node)

    # Define edges (linear pipeline)
    workflow.set_entry_point("scanner")
    workflow.add_edge("scanner", "analyst")
    workflow.add_edge("analyst", "risk")
    workflow.add_edge("risk", "execution")
    workflow.add_edge("execution", "portfolio_review")
    workflow.add_edge("portfolio_review", END)

    return workflow.compile()


# ---- Run Functions ----

def _validate_openai_key():
    """Check if OpenAI API key is configured before running pipeline"""
    key = settings.openai.api_key
    if not key or key == "your_openai_api_key_here" or len(key) < 10:
        raise ValueError(
            "OpenAI API key not configured! "
            "Please set your key in the sidebar under 'OpenAI Settings' or in the .env file."
        )


def run_full_pipeline() -> TradingState:
    """Run the complete trading pipeline"""
    _validate_openai_key()

    logger.info("=" * 80)
    logger.info("ALPHA-PRIME TRADING PIPELINE - STARTING")
    logger.info("=" * 80)

    pipeline = build_trading_pipeline()
    initial_state = create_initial_state()

    try:
        result = pipeline.invoke(initial_state)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        log_agent_activity("pipeline", "complete", "Full pipeline completed successfully")
        return result
    except Exception as e:
        logger.error(f"PIPELINE ERROR: {e}")
        log_agent_activity("pipeline", "error", str(e))
        initial_state["error"] = str(e)
        return initial_state


def run_scanner_only() -> str:
    """Run just the market scanner"""
    _validate_openai_key()
    state = create_initial_state()
    state = scanner_node(state)
    return state["scanner_output"]


def run_analysis(symbols: list) -> str:
    """Run analysis on specific symbols"""
    _validate_openai_key()
    prompt = get_analyst_prompt()
    user_msg = f"""Run detailed technical analysis on these specific stocks: {', '.join(symbols)}

For each stock:
1. Get current price
2. Run full technical analysis
3. Run intraday analysis
4. Provide BUY/SELL/HOLD recommendation with entry, stop-loss, and target
5. Save the signals"""

    return run_agent_node(
        create_initial_state(), prompt, user_msg,
        MARKET_ANALYSIS_TOOLS + SIGNAL_TOOLS,
        "technical_analyst",
        max_iterations=15
    )


def run_portfolio_review() -> str:
    """Run portfolio review only"""
    _validate_openai_key()
    state = create_initial_state()
    state = portfolio_review_node(state)
    return state["portfolio_output"]
