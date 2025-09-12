"""
Configuration Management Module

Centralized configuration with environment-based settings,
validation, and type safety following 2025 best practices.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass(frozen=True)
class ApiConfig:
    """API configuration with immutable settings"""
    sandbox_key: str
    production_key: str
    sandbox_base_url: str = "https://api.sandbox.competitions.recall.network"
    production_base_url: str = "https://api.competitions.recall.network"
    timeout: int = 30
    max_retries: int = 3


@dataclass(frozen=True)
class TradingConfig:
    """Trading strategy configuration"""
    default_amount: str = "1000"
    max_position_size: float = 0.25  # 25% of portfolio
    stop_loss_threshold: float = 0.05  # 5% stop loss
    take_profit_threshold: float = 0.10  # 10% take profit
    trade_interval_seconds: int = 60


@dataclass(frozen=True)
class TokenConfig:
    """Blockchain token configurations"""
    usdc_address: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    weth_address: str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    supported_chains: tuple = ("evm", "eth")


class Config:
    """
    Centralized configuration manager with validation

    Follows singleton pattern with environment-based configuration.
    Validates all settings on initialization.
    """

    _instance: Optional['Config'] = None

    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._validate_environment()

        # Initialize configuration sections
        self.api = ApiConfig(
            sandbox_key=self._get_required_env("RECALL_SANDBOX_API_KEY"),
            production_key=self._get_required_env("RECALL_PRODUCTION_API_KEY")
        )

        self.trading = TradingConfig(
            default_amount=os.getenv("RECALL_DEFAULT_AMOUNT", "100"),
            trade_interval_seconds=int(os.getenv("TRADE_INTERVAL_SECONDS", "60"))
        )

        self.tokens = TokenConfig()

        # Runtime settings
        self.use_production = os.getenv("RECALL_USE_PRODUCTION", "False").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_file = os.getenv("LOG_FILE", "recall_agent.log")

    def _get_required_env(self, key: str) -> str:
        """Get required environment variable with validation"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not found")
        return value

    def _validate_environment(self) -> None:
        """Validate critical environment variables"""
        required_vars = [
            "RECALL_SANDBOX_API_KEY",
            "RECALL_PRODUCTION_API_KEY"
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    @property
    def current_api_key(self) -> str:
        """Get current API key based on environment"""
        return self.api.production_key if self.use_production else self.api.sandbox_key

    @property
    def current_base_url(self) -> str:
        """Get current base URL based on environment"""
        return self.api.production_base_url if self.use_production else self.api.sandbox_base_url

    @property
    def environment_name(self) -> str:
        """Get current environment name"""
        return "PRODUCTION" if self.use_production else "SANDBOX"

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary for logging"""
        return {
            "environment": self.environment_name,
            "base_url": self.current_base_url,
            "trading_config": {
                "default_amount": self.trading.default_amount,
                "max_position_size": self.trading.max_position_size,
                "trade_interval": self.trading.trade_interval_seconds
            },
            "api_config": {
                "timeout": self.api.timeout,
                "max_retries": self.api.max_retries
            }
        }
# === Portfolio Manager settings (Recall Portfolio Tutorial compatible) ===
# Целевые доли портфеля по символам (сумма ≈ 1.0). Можно править под себя.
PORTFOLIO_TARGETS = {
    "USDC": 0.70,   # кэш-буфер
    "USDbC": 0.10,  # второй стейбл (пример)
    "WETH": 0.20,   # риск-ассет
}

# Допустимое отклонение долей (напр., 2%); если больше — запускаем ребаланс.
PORTFOLIO_DRIFT_THRESHOLD = 0.02
