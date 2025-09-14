"""
Configuration Module

Enterprise configuration management with environment-based settings,
comprehensive validation, and Claude AI integration support.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from utils import Logger

# Load environment variables
load_dotenv()


@dataclass
class TokenAddresses:
    """Token contract addresses for different networks"""
    usdc_address: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # Solana USDC
    weth_address: str = "So11111111111111111111111111111111111111112"   # Solana SOL
    wbtc_address: str = "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E"   # Solana BTC


@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    trade_interval_seconds: int = 5  # 5 minutes
    max_daily_trades: int = 5000
    max_trade_amount_usd: float = 1000.0
    min_trade_amount_usd: float = 10.0
    default_slippage_tolerance: float = 0.5  # 0.5%
    risk_management_enabled: bool = True
    auto_rebalance_enabled: bool = True
    rebalance_threshold: float = 0.02  # 2%

    # 🔥 ДОБАВЛЯЕМ НЕДОСТАЮЩИЕ ПОЛЯ
    default_amount: float = 10.0  # Для обратной совместимости

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Ensure default_amount matches min_trade_amount_usd
        self.default_amount = self.min_trade_amount_usd


@dataclass
class AIConfig:
    """AI/LLM integration configuration"""
    enabled: bool = True
    provider: str = "claude"  # "claude", "openai", "none"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1500
    temperature: float = 0.3
    min_confidence_threshold: float = 0.7
    rate_limit_seconds: int = 1
    fallback_to_simple_strategy: bool = True


class Config:
    """
    Central configuration management for the Recall trading bot

    Manages environment variables, API endpoints, trading parameters,
    and Claude AI integration settings.
    """

    def __init__(self):
        self.logger = Logger.get_logger("Config")

        # Environment detection
        self.use_production = self.get_env_var("RECALL_USE_PRODUCTION", "false").lower() == "true"
        self.environment_name = "production" if self.use_production else "SANDBOX"

        # API Configuration
        self.sandbox_api_key = self.get_env_var("RECALL_SANDBOX_API_KEY")
        self.production_api_key = self.get_env_var("RECALL_PRODUCTION_API_KEY")

        # Claude AI Configuration
        self.claude_api_key = self.get_env_var("CLAUDE_API_KEY")

        # 🔥 ПРОВЕРКА: если ключ есть но невалидный, логируем проблему
        if self.claude_api_key:
            if not self.claude_api_key.startswith(('sk-ant-', 'sk-')):
                self.logger.warning(f"⚠️ CLAUDE_API_KEY выглядит неправильно: {self.claude_api_key[:10]}...")
            else:
                self.logger.info("✅ CLAUDE_API_KEY найден и выглядит корректно")
        else:
            self.logger.warning("⚠️ CLAUDE_API_KEY не найден в переменных окружения")

        # OpenAI Configuration (fallback)
        self.openai_api_key = self.get_env_var("OPENAI_API_KEY")

        # API Endpoints
        self.sandbox_base_url = "https://api.sandbox.competitions.recall.network"
        self.production_base_url = "https://api.competitions.recall.network"

        # Component configurations
        self.tokens = TokenAddresses()
        self.trading = TradingConfig()
        self.ai = AIConfig()

        # Validate configuration
        self._validate_config()

        self.logger.info(
            "Configuration initialized",
            environment=self.environment_name,
            ai_provider=self.ai.provider,
            ai_enabled=self.ai.enabled,
            has_claude_key=bool(self.claude_api_key),
            has_openai_key=bool(self.openai_api_key)
        )

    @property
    def current_api_key(self) -> str:
        """Get the current API key based on environment"""
        if self.use_production:
            if not self.production_api_key:
                raise ValueError("RECALL_PRODUCTION_API_KEY не установлен для production окружения")
            return self.production_api_key
        else:
            if not self.sandbox_api_key:
                raise ValueError("RECALL_SANDBOX_API_KEY не установлен для sandbox окружения")
            return self.sandbox_api_key

    @property
    def current_base_url(self) -> str:
        """Get the current base URL based on environment"""
        return self.production_base_url if self.use_production else self.sandbox_base_url

    def get_env_var(self, var_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Safely get environment variable

        Args:
            var_name: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value or default
        """
        value = os.getenv(var_name, default)
        return value.strip() if value else value

    def _validate_config(self) -> None:
        """Validate configuration and set up AI provider"""
        # Validate Recall API keys
        if not self.sandbox_api_key and not self.production_api_key:
            raise ValueError("Не найден ни RECALL_SANDBOX_API_KEY, ни RECALL_PRODUCTION_API_KEY")

        # Configure AI provider с детальной диагностикой
        if self.claude_api_key:
            # 🔥 ПРОВЕРКА: если ключ есть но невалидный, логируем проблему
            if not self.claude_api_key.startswith(('sk-ant-', 'sk-')):
                self.logger.warning(f"⚠️ CLAUDE_API_KEY выглядит неправильно: {self.claude_api_key[:10]}...")
            else:
                self.logger.info("✅ CLAUDE_API_KEY найден и выглядит корректно")

            self.ai.provider = "claude"
            self.ai.enabled = True
            self.ai.model = "claude-sonnet-4-20250514"
            self.logger.info("✅ Claude AI настроен как основной LLM провайдер")

        elif self.openai_api_key:
            self.ai.provider = "openai"
            self.ai.enabled = True
            self.ai.model = "gpt-4o-mini"
            self.logger.warning("⚠️ Claude API недоступен, используем OpenAI как fallback")

        else:
            self.ai.provider = "none"
            self.ai.enabled = False
            self.logger.warning("⚠️ Нет доступных LLM API ключей! Бот будет работать без AI анализа.")

        # Validate trading parameters
        if self.trading.max_trade_amount_usd <= self.trading.min_trade_amount_usd:
            raise ValueError("max_trade_amount_usd должен быть больше min_trade_amount_usd")

    def get_ai_api_key(self) -> Optional[str]:
        """Get the current AI API key based on provider"""
        if self.ai.provider == "claude":
            return self.claude_api_key
        elif self.ai.provider == "openai":
            return self.openai_api_key
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for logging

        Returns:
            Dict with configuration (sensitive data masked)
        """
        return {
            "environment": self.environment_name,
            "base_url": self.current_base_url,
            "has_recall_key": bool(self.current_api_key),
            "ai_provider": self.ai.provider,
            "ai_enabled": self.ai.enabled,
            "ai_model": self.ai.model,
            "trading": {
                "interval_seconds": self.trading.trade_interval_seconds,
                "max_daily_trades": self.trading.max_daily_trades,
                "max_amount_usd": self.trading.max_trade_amount_usd,
                "min_amount_usd": self.trading.min_trade_amount_usd,
                "auto_rebalance": self.trading.auto_rebalance_enabled,
            },
            "tokens": {
                "usdc": self.tokens.usdc_address[:10] + "...",
                "weth": self.tokens.weth_address[:10] + "...",
                "wbtc": self.tokens.wbtc_address[:10] + "...",
            }
        }

    def is_ai_available(self) -> bool:
        """Check if AI provider is available and configured"""
        return self.ai.enabled and self.get_ai_api_key() is not None

    def get_strategy_config(self) -> Dict[str, Any]:
        """Get configuration for strategy selection"""
        return {
            "ai_enabled": self.is_ai_available(),
            "ai_provider": self.ai.provider,
            "fallback_enabled": self.ai.fallback_to_simple_strategy,
            "confidence_threshold": self.ai.min_confidence_threshold,
        }