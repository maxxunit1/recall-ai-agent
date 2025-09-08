"""
Strategy Engine Module

Advanced trading strategy implementation with ML-ready architecture,
risk management, and extensible strategy patterns.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

from config import Config
from utils import Logger, PerformanceTracker
from data_handler import DataHandler, PortfolioData, TokenPrice


class TradingAction(Enum):
    """Trading action enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class TradingSignal:
    """Trading signal with comprehensive metadata"""
    action: TradingAction
    from_token: str
    to_token: str
    amount: str
    confidence: float  # 0.0 to 1.0
    risk_level: RiskLevel
    reasoning: str
    timestamp: float
    strategy_name: str
    expected_return: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class MarketData:
    """Market data container for strategy analysis"""
    portfolio: Optional[PortfolioData]
    token_prices: Dict[str, TokenPrice]
    timestamp: float
    market_trend: Optional[str] = None


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies

    Provides framework for implementing various trading strategies
    with consistent interface and risk management.
    """

    def __init__(self, config: Config, data_handler: DataHandler):
        self.config = config
        self.data_handler = data_handler
        self.logger = Logger.get_logger(f"Strategy.{self.__class__.__name__}")
        self.performance_tracker = PerformanceTracker(self.logger)

        # Strategy state
        self.last_signal_time = 0.0
        self.position_history: List[TradingSignal] = []
        self.strategy_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0
        }

        self.logger.info(f"Strategy {self.__class__.__name__} initialized")

    @abstractmethod
    def analyze_market(self, market_data: MarketData) -> TradingSignal:
        """
        Analyze market conditions and generate trading signal

        Args:
            market_data: Current market data

        Returns:
            TradingSignal: Generated trading signal
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name identifier"""
        pass

    def validate_signal(self, signal: TradingSignal, market_data: MarketData) -> bool:
        """
        Validate trading signal against risk parameters

        Args:
            signal: Trading signal to validate
            market_data: Current market data

        Returns:
            bool: True if signal is valid
        """
        # Time-based validation
        if signal.timestamp - self.last_signal_time < self.config.trading.trade_interval_seconds:
            self.logger.debug("Signal rejected: Too frequent trading")
            return False

        # Confidence threshold
        if signal.confidence < 0.6:  # Minimum 60% confidence
            self.logger.debug("Signal rejected: Low confidence", confidence=signal.confidence)
            return False

        # Risk level validation
        if signal.risk_level == RiskLevel.CRITICAL:
            self.logger.warning("Signal rejected: Critical risk level")
            return False

        # Position size validation
        if market_data.portfolio:
            try:
                amount_float = float(signal.amount)
                portfolio_value = market_data.portfolio.total_value
                position_ratio = amount_float / portfolio_value

                if position_ratio > self.config.trading.max_position_size:
                    self.logger.warning(
                        "Signal rejected: Position too large",
                        position_ratio=position_ratio,
                        max_allowed=self.config.trading.max_position_size
                    )
                    return False
            except (ValueError, ZeroDivisionError) as e:
                self.logger.error("Position size validation failed", error=e)
                return False

        return True

    def update_metrics(self, signal: TradingSignal, success: bool, actual_return: float = 0.0):
        """Update strategy performance metrics"""
        self.strategy_metrics["total_trades"] += 1
        if success:
            self.strategy_metrics["successful_trades"] += 1

        self.strategy_metrics["total_return"] += actual_return
        self.strategy_metrics["win_rate"] = (
                self.strategy_metrics["successful_trades"] / self.strategy_metrics["total_trades"]
        )

        self.position_history.append(signal)

        self.logger.info(
            "Strategy metrics updated",
            strategy=self.get_strategy_name(),
            total_trades=self.strategy_metrics["total_trades"],
            win_rate=self.strategy_metrics["win_rate"],
            total_return=self.strategy_metrics["total_return"]
        )


class SimpleThresholdStrategy(BaseStrategy):
    """
    Simple threshold-based trading strategy

    Implements basic buy/sell logic based on price thresholds
    with risk management and position sizing.
    """

    def __init__(self, config: Config, data_handler: DataHandler):
        super().__init__(config, data_handler)

        # Strategy parameters
        self.buy_threshold = 95000.0  # USDC price threshold for buying
        self.sell_threshold = 105000.0  # USDC price threshold for selling
        self.confidence_base = 0.7  # Base confidence level

        self.logger.info(
            "SimpleThresholdStrategy configured",
            buy_threshold=self.buy_threshold,
            sell_threshold=self.sell_threshold
        )

    def get_strategy_name(self) -> str:
        return "SimpleThreshold"

    def analyze_market(self, market_data: MarketData) -> TradingSignal:
        """
        Analyze market using simple threshold logic

        Args:
            market_data: Current market data

        Returns:
            TradingSignal: Generated trading signal
        """
        operation_id = f"analyze_market_{int(time.time())}"
        self.performance_tracker.start_operation(operation_id)

        try:
            # Get USDC and WETH prices
            usdc_address = self.config.tokens.usdc_address
            weth_address = self.config.tokens.weth_address

            current_price = None
            if usdc_address in market_data.token_prices:
                current_price = market_data.token_prices[usdc_address].price
            elif weth_address in market_data.token_prices:
                # Use WETH price for demonstration
                current_price = market_data.token_prices[weth_address].price

            if current_price is None:
                self.logger.warning("No price data available for analysis")
                return self._create_hold_signal("No price data available")

            # Simple threshold logic
            if current_price < self.buy_threshold:
                return self._create_buy_signal(current_price, market_data)
            elif current_price > self.sell_threshold:
                return self._create_sell_signal(current_price, market_data)
            else:
                return self._create_hold_signal(f"Price {current_price} within threshold range")

        finally:
            self.performance_tracker.end_operation(operation_id, success=True)

    def _create_buy_signal(self, current_price: float, market_data: MarketData) -> TradingSignal:
        """Create buy trading signal"""
        price_deviation = (self.buy_threshold - current_price) / self.buy_threshold
        confidence = min(self.confidence_base + price_deviation, 1.0)

        # Calculate position size based on portfolio
        amount = self.config.trading.default_amount
        if market_data.portfolio:
            # Use percentage of portfolio value
            portfolio_value = market_data.portfolio.total_value
            max_position_value = portfolio_value * self.config.trading.max_position_size
            amount = str(min(float(amount), max_position_value))

        return TradingSignal(
            action=TradingAction.BUY,
            from_token=self.config.tokens.usdc_address,
            to_token=self.config.tokens.weth_address,
            amount=amount,
            confidence=confidence,
            risk_level=self._assess_risk(confidence),
            reasoning=f"Price {current_price} below buy threshold {self.buy_threshold}",
            timestamp=time.time(),
            strategy_name=self.get_strategy_name(),
            expected_return=price_deviation * 100,  # Expected % return
            stop_loss=current_price * (1 - self.config.trading.stop_loss_threshold),
            take_profit=current_price * (1 + self.config.trading.take_profit_threshold)
        )

    def _create_sell_signal(self, current_price: float, market_data: MarketData) -> TradingSignal:
        """Create sell trading signal"""
        price_deviation = (current_price - self.sell_threshold) / self.sell_threshold
        confidence = min(self.confidence_base + price_deviation, 1.0)

        # Calculate sell amount based on current holdings
        amount = self.config.trading.default_amount
        if market_data.portfolio:
            # Find WETH holdings
            weth_holdings = next(
                (token for token in market_data.portfolio.tokens
                 if token.get("token") == self.config.tokens.weth_address),
                None
            )
            if weth_holdings:
                # Sell portion of holdings
                max_sell = weth_holdings.get("amount", 0) * 0.5  # Sell up to 50%
                amount = str(min(float(amount), max_sell))

        return TradingSignal(
            action=TradingAction.SELL,
            from_token=self.config.tokens.weth_address,
            to_token=self.config.tokens.usdc_address,
            amount=amount,
            confidence=confidence,
            risk_level=self._assess_risk(confidence),
            reasoning=f"Price {current_price} above sell threshold {self.sell_threshold}",
            timestamp=time.time(),
            strategy_name=self.get_strategy_name(),
            expected_return=price_deviation * 100
        )

    def _create_hold_signal(self, reasoning: str) -> TradingSignal:
        """Create hold trading signal"""
        return TradingSignal(
            action=TradingAction.HOLD,
            from_token="",
            to_token="",
            amount="0",
            confidence=0.8,  # High confidence in holding
            risk_level=RiskLevel.LOW,
            reasoning=reasoning,
            timestamp=time.time(),
            strategy_name=self.get_strategy_name()
        )

    def _assess_risk(self, confidence: float) -> RiskLevel:
        """Assess risk level based on confidence"""
        if confidence >= 0.9:
            return RiskLevel.LOW
        elif confidence >= 0.75:
            return RiskLevel.MEDIUM
        elif confidence >= 0.6:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


class StrategyEngine:
    """
    Main strategy engine coordinating multiple strategies

    Manages strategy execution, signal validation, and performance tracking
    across multiple trading strategies.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.data_handler = DataHandler(self.config)
        self.logger = Logger.get_logger("StrategyEngine")

        # Initialize strategies
        self.strategies: List[BaseStrategy] = [
            SimpleThresholdStrategy(self.config, self.data_handler)
        ]

        self.active_strategy = self.strategies[0]  # Default strategy

        self.logger.info(
            "StrategyEngine initialized",
            total_strategies=len(self.strategies),
            active_strategy=self.active_strategy.get_strategy_name()
        )

    def get_market_data(self) -> Optional[MarketData]:
        """
        Gather current market data for analysis

        Returns:
            Optional[MarketData]: Market data or None if failed
        """
        try:
            # Get portfolio data
            portfolio_success, portfolio = self.data_handler.get_portfolio()
            if not portfolio_success:
                self.logger.warning("Failed to get portfolio data")
                portfolio = None

            # Get token prices
            token_prices = {}
            for token_address in [self.config.tokens.usdc_address, self.config.tokens.weth_address]:
                price_success, price_data = self.data_handler.get_token_price(token_address)
                if price_success and price_data:
                    token_prices[token_address] = price_data

            if not token_prices:
                self.logger.error("Failed to get any token prices")
                return None

            return MarketData(
                portfolio=portfolio,
                token_prices=token_prices,
                timestamp=time.time()
            )

        except Exception as e:
            self.logger.error("Failed to gather market data", error=e)
            return None

    def generate_signal(self) -> Optional[TradingSignal]:
        """
        Generate trading signal using active strategy

        Returns:
            Optional[TradingSignal]: Generated signal or None if failed
        """
        market_data = self.get_market_data()
        if not market_data:
            return None

        try:
            signal = self.active_strategy.analyze_market(market_data)

            if self.active_strategy.validate_signal(signal, market_data):
                self.logger.info(
                    "Valid trading signal generated",
                    action=signal.action.value,
                    confidence=signal.confidence,
                    strategy=signal.strategy_name
                )
                return signal
            else:
                self.logger.debug("Generated signal failed validation")
                return None

        except Exception as e:
            self.logger.error("Failed to generate trading signal", error=e)
            return None

    def set_active_strategy(self, strategy_name: str) -> bool:
        """
        Set active trading strategy

        Args:
            strategy_name: Name of strategy to activate

        Returns:
            bool: True if strategy was set successfully
        """
        for strategy in self.strategies:
            if strategy.get_strategy_name() == strategy_name:
                self.active_strategy = strategy
                self.logger.info("Active strategy changed", new_strategy=strategy_name)
                return True

        self.logger.error("Strategy not found", strategy_name=strategy_name)
        return False

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all strategies"""
        return {
            strategy.get_strategy_name(): strategy.strategy_metrics
            for strategy in self.strategies
        }
