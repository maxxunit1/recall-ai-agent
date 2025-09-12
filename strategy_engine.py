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
            except (ValueError, TypeError):
                self.logger.error("Signal rejected: Invalid amount format")
                return False

        return True

    def update_performance(self, signal: TradingSignal, success: bool, actual_return: Optional[float] = None):
        """Update strategy performance metrics"""
        self.strategy_metrics["total_trades"] += 1

        if success:
            self.strategy_metrics["successful_trades"] += 1
            if actual_return:
                self.strategy_metrics["total_return"] += actual_return

        # Update win rate
        self.strategy_metrics["win_rate"] = (
                self.strategy_metrics["successful_trades"] / self.strategy_metrics["total_trades"]
        )

        # Update last signal time
        self.last_signal_time = signal.timestamp


class SimpleThresholdStrategy(BaseStrategy):
    """
    Simple threshold-based trading strategy

    Based on Trading Guide examples: buys low, sells high with fixed thresholds
    """

    def __init__(self, config: Config, data_handler: DataHandler):
        super().__init__(config, data_handler)

        # Strategy parameters - USDC typically trades around 1.0
        self.buy_threshold = 0.9990  # Buy WETH when USDC cheap (below 0.9990)
        self.sell_threshold = 1.0010  # Sell WETH when USDC expensive (above 1.0010)

        self.logger.info("SimpleThresholdStrategy configured",
                         buy_threshold=self.buy_threshold,
                         sell_threshold=self.sell_threshold)

    def get_strategy_name(self) -> str:
        return "SimpleThreshold"

    def analyze_market(self, market_data: MarketData) -> TradingSignal:
        """
        Analyze market and generate threshold-based signal

        Logic:
        - If USDC price < buy_threshold -> BUY WETH with USDC
        - If USDC price > sell_threshold -> SELL WETH for USDC
        - Otherwise -> HOLD
        """
        # Get USDC price for decision making
        usdc_price = None
        for token_addr, price_data in market_data.token_prices.items():
            if token_addr == self.config.tokens.usdc_address or "usdc" in token_addr.lower():
                usdc_price = price_data.price if hasattr(price_data, 'price') else price_data
                break

        if usdc_price is None:
            return self._create_hold_signal("No USDC price available")

        current_price = float(usdc_price)  # Use actual USDC price (around 1.0)

        # Calculate confidence based on distance from threshold
        if current_price < self.buy_threshold:
            # BUY signal
            price_deviation = (self.buy_threshold - current_price) / self.buy_threshold
            confidence = min(0.9, 0.6 + price_deviation * 2)  # 60-90% confidence

            amount = self.config.trading.default_amount

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
                expected_return=price_deviation * 100
            )

        elif current_price > self.sell_threshold:
            # SELL signal
            price_deviation = (current_price - self.sell_threshold) / self.sell_threshold
            confidence = min(0.9, 0.6 + price_deviation * 2)

            # Calculate amount to sell (need to check WETH balance)
            amount = self.config.trading.default_amount
            if market_data.portfolio and market_data.portfolio.tokens:
                for token in market_data.portfolio.tokens:
                    if token.get("symbol") == "WETH" or token.get("token") == self.config.tokens.weth_address:
                        # Use available WETH balance (with some buffer)
                        available = float(token.get("amount", 0))
                        amount = str(min(float(self.config.trading.default_amount), available * 0.8))
                        break

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
        else:
            # HOLD signal
            return self._create_hold_signal(
                f"Price {current_price} between thresholds {self.buy_threshold}-{self.sell_threshold}"
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


class PortfolioManager:
    """
    Portfolio Manager based on official Portfolio Manager Tutorial

    Uses CoinGecko for prices and simple balance format from /api/balance
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger.get_logger("PortfolioManager")

        # Token configuration (exactly from tutorial)
        self.TOKEN_MAP = {
            "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
            "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
        }

        self.DECIMALS = {
            "USDC": 6,
            "WETH": 18,
            "WBTC": 8
        }

        self.COINGECKO_IDS = {
            "USDC": "usd-coin",
            "WETH": "weth",
            "WBTC": "wrapped-bitcoin"
        }

        # Portfolio targets (from portfolio_config.json in tutorial)
        self.targets = {
            "USDC": 0.60,  # 60%
            "WETH": 0.30,  # 30%
            "WBTC": 0.10  # 10%
        }

        self.DRIFT_THRESHOLD = 0.02  # 2% as in tutorial

        self.logger.info("PortfolioManager initialized (tutorial mode)",
                         targets=self.targets,
                         drift_threshold=self.DRIFT_THRESHOLD)

    def fetch_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch prices from CoinGecko (exactly from tutorial)"""
        import requests

        ids = ",".join(self.COINGECKO_IDS[sym] for sym in symbols if sym in self.COINGECKO_IDS)
        if not ids:
            self.logger.error("No valid CoinGecko IDs found", symbols=symbols)
            return {}

        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": ids, "vs_currencies": "usd"},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()

            result = {}
            for sym in symbols:
                if sym in self.COINGECKO_IDS:
                    cg_id = self.COINGECKO_IDS[sym]
                    if cg_id in data:
                        result[sym] = data[cg_id]["usd"]

            self.logger.info("CoinGecko prices fetched", prices=result)
            return result

        except Exception as e:
            self.logger.error("Failed to fetch CoinGecko prices", error=e)
            return {}

    def fetch_holdings(self) -> dict[str, float]:
        """Fetch holdings from Recall /api/balance (tutorial format)"""
        try:
            # Fix: create DataHandler instance instead of using config.data_handler
            from data_handler import DataHandler
            data_handler = DataHandler(self.config)
            success, balances_list = data_handler.get_balances()
            if not success or not balances_list:
                self.logger.error("Failed to fetch holdings")
                return {}

            # Convert list format to simple dict format
            balances = {}
            for item in balances_list:
                symbol = item.get("symbol", "UNKNOWN")
                amount = float(item.get("amount", 0))
                balances[symbol] = amount
            if not success or not balances:
                self.logger.error("Failed to fetch holdings")
                return {}

            # Filter to our supported tokens
            holdings = {}
            for symbol in self.targets.keys():
                holdings[symbol] = balances.get(symbol, 0.0)

            self.logger.info("Holdings fetched", holdings=holdings)
            return holdings

        except Exception as e:
            self.logger.error("Failed to fetch holdings", error=e)
            return {}

    def compute_orders(self, targets: dict, prices: dict, holdings: dict) -> list[dict]:
        """Compute rebalancing orders (exactly from tutorial)"""
        total_value = sum(holdings.get(s, 0) * prices.get(s, 0) for s in targets)

        if total_value == 0:
            raise ValueError("No balances found; fund your sandbox wallet first.")

        overweight, underweight = [], []
        for sym, weight in targets.items():
            current_val = holdings.get(sym, 0) * prices.get(sym, 0)
            target_val = total_value * weight
            drift_pct = (current_val - target_val) / total_value

            if abs(drift_pct) >= self.DRIFT_THRESHOLD:
                delta_val = abs(target_val - current_val)
                token_amt = delta_val / prices.get(sym, 0)
                side = "sell" if drift_pct > 0 else "buy"

                self.logger.info(f"Rebalance needed: {sym} {side} {token_amt:.6f}")

                # Convert Portfolio Manager format to Trading Guide format
                # Special handling for USDC: when selling USDC, we buy other tokens
                if sym == "USDC":
                    # USDC sell = buy other tokens, skip this - we'll handle in the buying phase
                    if side == "sell":
                        continue
                    from_token = self.TOKEN_MAP["USDC"]
                    to_token = self.TOKEN_MAP[sym]
                else:
                    # Non-USDC tokens: sell to USDC or buy with USDC
                    from_token, to_token = (
                        (self.TOKEN_MAP[sym], self.TOKEN_MAP["USDC"]) if side == "sell"
                        else (self.TOKEN_MAP["USDC"], self.TOKEN_MAP[sym])
                    )

                (overweight if side == "sell" else underweight).append({
                    "from_token": from_token,
                    "to_token": to_token,
                    "amount": str(token_amt),
                    "reasoning": f"Rebalance {sym}: target allocation drift {abs(drift_pct) * 100:.1f}%"
                })

        # Execute sells first so we have USDC to fund buys
        self.logger.info("Rebalance orders computed", orders_count=len(overweight + underweight))
        return overweight + underweight

    def plan_rebalance(self, portfolio=None) -> list[dict]:
        """Main rebalancing function (tutorial interface)"""
        try:
            # 1. Fetch current prices from CoinGecko
            symbols = list(self.targets.keys())
            prices = self.fetch_prices(symbols)
            if not prices:
                self.logger.error("No prices available for rebalancing")
                return []

            # 2. Fetch current holdings
            holdings = self.fetch_holdings()
            if not holdings:
                self.logger.error("No holdings available for rebalancing")
                return []

            # 3. Compute orders
            orders = self.compute_orders(self.targets, prices, holdings)

            self.logger.info("Rebalance plan created (tutorial mode)",
                             orders_count=len(orders),
                             total_value=sum(holdings.get(s, 0) * prices.get(s, 0) for s in symbols))

            return orders

        except Exception as e:
            self.logger.error("Failed to plan rebalance", error=e)
            return []

    def analyze_portfolio(self, portfolio=None) -> dict:
        """Analyze portfolio (simplified for tutorial compatibility)"""
        try:
            symbols = list(self.targets.keys())
            prices = self.fetch_prices(symbols)
            holdings = self.fetch_holdings()

            if not prices or not holdings:
                return {"error": "No data available"}

            total_value = sum(holdings.get(s, 0) * prices.get(s, 0) for s in symbols)

            current_allocations = {}
            deviations = {}
            max_deviation = 0.0

            for symbol in symbols:
                current_val = holdings.get(symbol, 0) * prices.get(symbol, 0)
                current_pct = current_val / total_value if total_value > 0 else 0
                target_pct = self.targets[symbol]
                deviation = current_pct - target_pct

                current_allocations[symbol] = current_pct
                deviations[symbol] = deviation
                max_deviation = max(max_deviation, abs(deviation))

            needs_rebalancing = max_deviation > self.DRIFT_THRESHOLD

            return {
                "total_value": total_value,
                "current_allocations": current_allocations,
                "target_allocations": self.targets,
                "deviations": deviations,
                "max_deviation": max_deviation,
                "needs_rebalancing": needs_rebalancing,
                "rebalance_threshold": self.DRIFT_THRESHOLD
            }

        except Exception as e:
            self.logger.error("Portfolio analysis failed", error=e)
            return {"error": str(e)}


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

            # Get token prices for major tokens
            token_prices = {}
            token_addresses = [self.config.tokens.usdc_address, self.config.tokens.weth_address]

            for token_address in token_addresses:
                price_success, price_data = self.data_handler.get_token_price(token_address)
                if price_success and price_data:
                    token_prices[token_address] = TokenPrice(
                        address=token_address,
                        price=price_data,
                        chain="evm",
                        specific_chain="eth"
                    )

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

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add new strategy to engine"""
        self.strategies.append(strategy)
        self.logger.info("Strategy added", strategy_name=strategy.get_strategy_name())

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return [strategy.get_strategy_name() for strategy in self.strategies]
