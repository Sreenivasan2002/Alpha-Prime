"""
Configuration Management for Alpha-Prime Trading System
Handles all environment variables, trading parameters, and system settings.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)


class GrowwConfig(BaseSettings):
    """Groww API Configuration"""
    api_key: str = Field(default="", alias="GROWW_API_KEY")
    secret_key: str = Field(default="", alias="GROWW_SECRET_KEY")

    class Config:
        env_prefix = ""
        extra = "ignore"


class OpenAIConfig(BaseSettings):
    """OpenAI Configuration"""
    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    temperature: float = Field(default=0.1, alias="OPENAI_TEMPERATURE")

    class Config:
        env_prefix = ""
        extra = "ignore"


class TradingConfig(BaseSettings):
    """Trading Parameters"""
    mode: str = Field(default="paper", alias="TRADING_MODE")  # paper or live
    max_daily_loss_pct: float = Field(default=2.0, alias="MAX_DAILY_LOSS_PCT")
    # Notional cap for any single position as a % of REAL capital (not margin
    # buying power). Bounds overnight-gap exposure on one name regardless of leverage.
    max_position_size_pct: float = Field(default=25.0, alias="MAX_POSITION_SIZE_PCT")
    max_open_positions: int = Field(default=3, alias="MAX_OPEN_POSITIONS")
    default_stop_loss_pct: float = Field(default=0.8, alias="DEFAULT_STOP_LOSS_PCT")
    default_target_pct: float = Field(default=1.2, alias="DEFAULT_TARGET_PCT")
    capital: float = Field(default=10000.0, alias="TRADING_CAPITAL")
    # Margin/leverage multiplier (e.g., 5x means Rs.10K capital = Rs.50K buying power)
    margin_multiplier: float = Field(default=5.0, alias="MARGIN_MULTIPLIER")
    # Stop taking new trades once daily P&L reaches this
    daily_profit_target_pct: float = Field(default=1.5, alias="DAILY_PROFIT_TARGET_PCT")
    # === Risk-based position sizing (enforced in CODE, not just the LLM prompt) ===
    # Max % of REAL capital to lose if a single trade hits its stop-loss.
    # This is the primary position-size control: quantity is sized so that
    # (entry - stop_loss) * quantity <= capital * risk_per_trade_pct/100.
    # With 0.5%, a stop-out costs ~Rs.5k on Rs.10L capital instead of Rs.17k+.
    risk_per_trade_pct: float = Field(default=0.5, alias="RISK_PER_TRADE_PCT")
    # === Trailing-stop / exit tuning (PositionMonitor) ===
    # Once a position is up this much, move the stop to breakeven (risk-free runner).
    breakeven_arm_pct: float = Field(default=0.6, alias="BREAKEVEN_ARM_PCT")
    # Don't start trailing the stop until the position is up at least this much.
    # Prevents winners from being knocked out at ~+0.5% before they can reach target.
    trail_arm_pct: float = Field(default=1.0, alias="TRAIL_ARM_PCT")
    # Once armed, trail the stop this far below the peak price.
    trailing_stop_pct: float = Field(default=1.0, alias="TRAILING_STOP_PCT")
    # Minimum signal strength (0-1) to consider a BUY
    # For intraday 5-min data, typical BUY strengths are 0.10-0.35
    # The engine already requires: signal=BUY + trend=BULLISH + volume_confirmed
    # So strength threshold is a secondary filter
    min_signal_strength: float = Field(default=0.10, alias="MIN_SIGNAL_STRENGTH")

    @property
    def effective_capital(self) -> float:
        """Capital * margin multiplier = actual buying power for intraday"""
        return self.capital * self.margin_multiplier

    class Config:
        env_prefix = ""
        extra = "ignore"


class DemoConfig(BaseSettings):
    """Public-demo mode.

    When enabled the app is a read-only showcase: it reads a committed
    snapshot database and every path that would spend money, touch the
    broker, call an LLM, or persist credentials is disabled. This is what
    runs on the public Streamlit Cloud deployment, where there are no API
    keys and any visitor can click any button.
    """
    enabled: bool = Field(default=False, alias="DEMO_MODE")

    class Config:
        env_prefix = ""
        extra = "ignore"


class MarketConfig(BaseSettings):
    """Indian Market Configuration"""
    exchange: str = "NSE"
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 30
    pre_open_start_hour: int = 9
    pre_open_start_minute: int = 0
    timezone: str = "Asia/Kolkata"

    # Don't place new BUY trades before this time (let morning volatility settle)
    # Format: hour and minute in IST. Default 10:00 AM.
    trading_start_hour: int = Field(default=10, alias="TRADING_START_HOUR")
    trading_start_minute: int = Field(default=0, alias="TRADING_START_MINUTE")
    # Stop placing new BUY trades after this time (need time for positions to play out)
    trading_end_hour: int = Field(default=14, alias="TRADING_END_HOUR")
    trading_end_minute: int = Field(default=30, alias="TRADING_END_MINUTE")

    # Agent run schedule
    analysis_interval_minutes: int = Field(default=15, alias="ANALYSIS_INTERVAL_MINUTES")

    class Config:
        env_prefix = ""
        extra = "ignore"


class Settings:
    """Master Settings Container"""

    def __init__(self):
        self.groww = GrowwConfig()
        self.openai = OpenAIConfig()
        self.trading = TradingConfig()
        self.market = MarketConfig()
        self.demo = DemoConfig()
        self.project_root = PROJECT_ROOT
        # Demo mode reads the committed read-only snapshot, never the live DB.
        if self.demo.enabled:
            self.db_path = PROJECT_ROOT / "data" / "demo" / "alpha_prime_demo.db"
        else:
            self.db_path = PROJECT_ROOT / "data" / "alpha_prime.db"
        self.log_path = PROJECT_ROOT / "logs"

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

    def update_groww_keys(self, api_key: str, secret_key: str):
        """Update Groww API keys at runtime (for UI key rotation)"""
        self._reject_if_demo("update broker credentials")
        self.groww.api_key = api_key
        self.groww.secret_key = secret_key
        # Also update os.environ so any re-reads pick it up
        os.environ["GROWW_API_KEY"] = api_key
        os.environ["GROWW_SECRET_KEY"] = secret_key

        # Persist to .env file
        self._update_env_file({"GROWW_API_KEY": api_key, "GROWW_SECRET_KEY": secret_key})

    def update_openai_settings(self, api_key: str, model: str = None):
        """Update OpenAI settings at runtime"""
        self._reject_if_demo("update OpenAI settings")
        self.openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        if model:
            self.openai.model = model
            os.environ["OPENAI_MODEL"] = model

        updates = {"OPENAI_API_KEY": api_key}
        if model:
            updates["OPENAI_MODEL"] = model
        self._update_env_file(updates)

    def _reject_if_demo(self, action: str):
        """Hard stop for credential writes on the public demo deployment.

        The UI disables these controls, but this is the backstop: no code
        path may persist a key when the app is publicly reachable.
        """
        if self.demo.enabled:
            raise PermissionError(
                f"DEMO_MODE is enabled - refusing to {action}. "
                "Run locally with DEMO_MODE=false for live configuration."
            )

    def _update_env_file(self, updates: dict):
        """Update specific keys in the .env file"""
        self._reject_if_demo("write to the .env file")
        env_path = self.project_root / ".env"
        lines = []
        if env_path.exists():
            with open(env_path, "r") as f:
                lines = f.readlines()

        new_lines = []
        found_keys = set()
        for line in lines:
            replaced = False
            for key, value in updates.items():
                if line.startswith(f"{key}="):
                    new_lines.append(f"{key}={value}\n")
                    found_keys.add(key)
                    replaced = True
                    break
            if not replaced:
                new_lines.append(line)

        # Add any keys not found in existing file
        for key, value in updates.items():
            if key not in found_keys:
                new_lines.append(f"{key}={value}\n")

        with open(env_path, "w") as f:
            f.writelines(new_lines)

    def is_configured(self) -> bool:
        """Check if minimum config is set"""
        return bool(self.groww.api_key and self.groww.secret_key and self.openai.api_key)


# Singleton
settings = Settings()
