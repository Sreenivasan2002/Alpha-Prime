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
    max_position_size_pct: float = Field(default=25.0, alias="MAX_POSITION_SIZE_PCT")
    max_open_positions: int = Field(default=3, alias="MAX_OPEN_POSITIONS")
    default_stop_loss_pct: float = Field(default=0.8, alias="DEFAULT_STOP_LOSS_PCT")
    default_target_pct: float = Field(default=1.2, alias="DEFAULT_TARGET_PCT")
    capital: float = Field(default=10000.0, alias="TRADING_CAPITAL")
    # Margin/leverage multiplier (e.g., 5x means Rs.10K capital = Rs.50K buying power)
    margin_multiplier: float = Field(default=5.0, alias="MARGIN_MULTIPLIER")
    # Stop taking new trades once daily P&L reaches this
    daily_profit_target_pct: float = Field(default=1.5, alias="DAILY_PROFIT_TARGET_PCT")
    # Minimum signal strength (0-1) to consider a BUY - higher = fewer but better trades
    min_signal_strength: float = Field(default=0.7, alias="MIN_SIGNAL_STRENGTH")

    @property
    def effective_capital(self) -> float:
        """Capital * margin multiplier = actual buying power for intraday"""
        return self.capital * self.margin_multiplier

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
        self.project_root = PROJECT_ROOT
        self.db_path = PROJECT_ROOT / "data" / "alpha_prime.db"
        self.log_path = PROJECT_ROOT / "logs"

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

    def update_groww_keys(self, api_key: str, secret_key: str):
        """Update Groww API keys at runtime (for UI key rotation)"""
        self.groww.api_key = api_key
        self.groww.secret_key = secret_key
        # Also update os.environ so any re-reads pick it up
        os.environ["GROWW_API_KEY"] = api_key
        os.environ["GROWW_SECRET_KEY"] = secret_key

        # Persist to .env file
        self._update_env_file({"GROWW_API_KEY": api_key, "GROWW_SECRET_KEY": secret_key})

    def update_openai_settings(self, api_key: str, model: str = None):
        """Update OpenAI settings at runtime"""
        self.openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        if model:
            self.openai.model = model
            os.environ["OPENAI_MODEL"] = model

        updates = {"OPENAI_API_KEY": api_key}
        if model:
            updates["OPENAI_MODEL"] = model
        self._update_env_file(updates)

    def _update_env_file(self, updates: dict):
        """Update specific keys in the .env file"""
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
