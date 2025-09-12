"""
Trade Executor Module

Enterprise-grade trade execution engine with comprehensive
error handling, transaction tracking, and performance analytics.
"""

import time
import json
import requests
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from config import Config
from utils import Logger, PerformanceTracker, retry_with_backoff
from data_handler import DataHandler, ApiResponse
from strategy_engine import TradingSignal, TradingAction


class ExecutionStatus(Enum):
    """Trade execution status enumeration"""
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class TradeResult:
    """Trade execution result container"""
    signal: TradingSignal
    status: ExecutionStatus
    transaction_id: Optional[str] = None
    executed_amount: Optional[float] = None
    executed_price: Optional[float] = None
    actual_return: Optional[float] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    slippage: Optional[float] = None
    gas_cost: Optional[float] = None


@dataclass
class ExecutionMetrics:
    """Trade execution performance metrics"""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_volume: float = 0.0
    total_fees: float = 0.0
    average_execution_time: float = 0.0
    success_rate: float = 0.0


class TradeExecutor:
    """
    Enterprise trade execution engine

    Provides robust trade execution with comprehensive error handling,
    transaction tracking, and performance monitoring.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.data_handler = DataHandler(self.config)
        self.logger = Logger.get_logger("TradeExecutor")
        self.performance_tracker = PerformanceTracker(self.logger)

        # Execution state
        self.trade_history: List[TradeResult] = []
        self.metrics = ExecutionMetrics()
        self.is_active = False

        self.logger.info(
            "TradeExecutor initialized",
            environment=self.config.environment_name
        )

    def start_execution(self) -> None:
        """Start trade execution engine"""
        self.is_active = True
        self.logger.info("Trade execution started")

    def stop_execution(self) -> None:
        """Stop trade execution engine"""
        self.is_active = False
        self.logger.info("Trade execution stopped")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def execute_trade(self, from_token: str, to_token: str, amount: str) -> TradeResult:
        """
        Execute trade via Recall API

        Based on Trading Guide: POST /api/trade/execute

        Args:
            from_token: Source token address or symbol (positional only)
            to_token: Target token address or symbol (positional only)
            amount: Amount to trade (positional only)

        Returns:
            TradeResult: Execution result with status and details
        """
        start_time = time.time()
        operation_id = f"trade_{int(start_time)}"

        self.performance_tracker.start_operation(operation_id)

        # Create dummy signal for result tracking
        dummy_signal = TradingSignal(
            action=TradingAction.BUY if from_token != self.config.tokens.weth_address else TradingAction.SELL,
            from_token=from_token,
            to_token=to_token,
            amount=str(amount),
            confidence=1.0,
            risk_level="LOW",
            reasoning="Rebalance trade execution",
            timestamp=start_time,
            strategy_name="DirectExecution"
        )

        try:
            # Validate execution conditions
            if not self._validate_execution_conditions(dummy_signal):
                result = TradeResult(
                    signal=dummy_signal,
                    status=ExecutionStatus.FAILED,
                    error_message="Execution conditions not met",
                    execution_time=time.time() - start_time
                )
                self._update_metrics(result)
                return result

            self.logger.info("Executing trade via API",
                             from_token=from_token,
                             to_token=to_token,
                             amount=amount)

            # Execute trade via API
            trade_data = {
                "fromToken": from_token,
                "toToken": to_token,
                "amount": str(amount),  # ✅ Правильное поле согласно доке
                "reason": "Verification trade for agent setup"
            }

            response = self._make_trade_request(trade_data, operation_id)
            result = self._process_execution_result(response, dummy_signal, start_time)

            # Update metrics and history
            self._update_metrics(result)
            self.trade_history.append(result)

            self.performance_tracker.end_operation(operation_id, success=(result.status == ExecutionStatus.EXECUTED))

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)

            # Enhanced nonce/reset error handling from Discord chat
            if "expected sequence" in error_message.lower() or "nonce" in error_message.lower():
                error_message += (" | Hint: Testnet reset detected. In your browser wallet "
                                  "(e.g. MetaMask), use 'Reset account' while connected to "
                                  "Recall Testnet, then re-fund test tokens and purchase credits.")

            self.logger.error("Trade execution failed",
                              error=error_message,
                              from_token=from_token,
                              to_token=to_token,
                              amount=amount,
                              execution_time=execution_time)

            result = TradeResult(
                signal=dummy_signal,
                status=ExecutionStatus.FAILED,
                error_message=error_message,
                execution_time=execution_time
            )

            self._update_metrics(result)
            self.trade_history.append(result)
            self.performance_tracker.end_operation(operation_id, success=False)

            return result

    def _make_trade_request(self, trade_data: Dict[str, Any], operation_id: str) -> ApiResponse:
        """
        Make direct trade request to Recall API

        Based on API Reference: POST /api/trade/execute
        Must send JSON directly without data wrapper
        """
        import requests

        url = f"{self.config.current_base_url}/api/trade/execute"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.current_api_key}",
            "User-Agent": "RecallAI-TradingAgent/2.0"
        }

        try:
            response = requests.post(
                url,
                json=trade_data,  # Прямая отправка JSON
                headers=headers,
                timeout=self.config.api.timeout
            )

            if response.status_code == 200:
                try:
                    response_data = response.json() if response.content else {}
                    return ApiResponse(success=True, data=response_data, status_code=200)
                except:
                    return ApiResponse(success=False, error="Invalid JSON", status_code=200)
            else:
                error_text = response.text if response.content else f"HTTP {response.status_code}"
                return ApiResponse(success=False, error=error_text, status_code=response.status_code)

        except Exception as e:
            return ApiResponse(success=False, error=str(e))

    def _process_execution_result(self, response: ApiResponse, signal: TradingSignal, start_time: float) -> TradeResult:
        """
        Process API response and create TradeResult

        Handles various response formats from Recall API
        """
        execution_time = time.time() - start_time

        if not response.success:
            error_message = response.error or "Unknown API error"

            # Enhanced nonce/reset error handling
            if "expected sequence" in error_message.lower() or "nonce" in error_message.lower():
                error_message += (" | Hint: Testnet reset detected. In your browser wallet "
                                  "(e.g. MetaMask), use 'Reset account' while connected to "
                                  "Recall Testnet, then re-fund test tokens and purchase credits.")

            return TradeResult(
                signal=signal,
                status=ExecutionStatus.FAILED,
                error_message=error_message,
                execution_time=execution_time
            )

        # Process successful response
        data = response.data or {}

        transaction_id = (data.get("id") or
                          data.get("transaction_id") or
                          data.get("transactionId") or
                          f"tx_{int(start_time)}")

        executed_amount = (data.get("fromAmount") or
                           data.get("executed_amount") or
                           data.get("executedAmount") or
                           float(signal.amount))

        executed_price = (data.get("price") or
                          data.get("executed_price") or
                          data.get("executedPrice"))

        self.logger.info("Trade executed successfully",
                         transaction_id=transaction_id,
                         executed_amount=executed_amount,
                         executed_price=executed_price,
                         execution_time=execution_time)

        return TradeResult(
            signal=signal,
            status=ExecutionStatus.EXECUTED,
            transaction_id=transaction_id,
            executed_amount=float(executed_amount) if executed_amount else None,
            executed_price=float(executed_price) if executed_price else None,
            execution_time=execution_time
        )

    def _validate_execution_conditions(self, signal: TradingSignal) -> bool:
        """
        Validate conditions before trade execution

        Args:
            signal: Trading signal to validate

        Returns:
            bool: True if conditions are valid for execution
        """
        # Check if trading is active
        if not self.is_active:
            self.logger.warning("Trade execution skipped: Executor not active")
            return False

        # Validate signal completeness
        if signal.action in [TradingAction.BUY, TradingAction.SELL]:
            if not all([signal.from_token, signal.to_token, signal.amount]):
                self.logger.error("Invalid signal: Missing required fields")
                return False

            try:
                amount_float = float(signal.amount)
                if amount_float <= 0:
                    self.logger.error("Invalid signal: Amount must be positive")
                    return False
            except (ValueError, TypeError):
                self.logger.error("Invalid signal: Amount must be numeric")
                return False

        return True

    def _update_metrics(self, result: TradeResult) -> None:
        """Update execution metrics with trade result"""
        self.metrics.total_trades += 1

        if result.status == ExecutionStatus.EXECUTED:
            self.metrics.successful_trades += 1
            if result.executed_amount:
                self.metrics.total_volume += result.executed_amount
        else:
            self.metrics.failed_trades += 1

        # Update success rate
        if self.metrics.total_trades > 0:
            self.metrics.success_rate = self.metrics.successful_trades / self.metrics.total_trades

        # Update average execution time
        if result.execution_time:
            current_avg = self.metrics.average_execution_time
            total_trades = self.metrics.total_trades
            self.metrics.average_execution_time = (
                    (current_avg * (total_trades - 1) + result.execution_time) / total_trades
            )

    def execute_trade_signal(self, signal: TradingSignal) -> TradeResult:
        """
        Execute trade from trading signal

        Args:
            signal: Trading signal from strategy engine

        Returns:
            TradeResult: Execution result
        """
        if signal.action == TradingAction.HOLD:
            # Create hold result
            return TradeResult(
                signal=signal,
                status=ExecutionStatus.CANCELLED,
                error_message="Hold signal - no trade executed",
                execution_time=0.0
            )

        return self.execute_trade(signal.from_token, signal.to_token, signal.amount)

    def get_execution_metrics(self) -> ExecutionMetrics:
        """Get current execution metrics"""
        return self.metrics

    def export_trade_history(self) -> List[Dict[str, Any]]:
        """
        Export trade history for analysis

        Returns:
            List of trade results as dictionaries
        """
        return [
            {
                "timestamp": result.signal.timestamp,
                "action": result.signal.action.value,
                "from_token": result.signal.from_token,
                "to_token": result.signal.to_token,
                "amount": result.signal.amount,
                "status": result.status.value,
                "transaction_id": result.transaction_id,
                "executed_amount": result.executed_amount,
                "executed_price": result.executed_price,
                "execution_time": result.execution_time,
                "error_message": result.error_message,
                "strategy": result.signal.strategy_name,
                "confidence": result.signal.confidence,
                "reasoning": getattr(result.signal, 'reasoning', 'No reasoning provided')
            }
            for result in self.trade_history
        ]

    def reset_metrics(self) -> None:
        """Reset execution metrics"""
        self.metrics = ExecutionMetrics()
        self.logger.info("Execution metrics reset")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary

        Returns:
            Dictionary with performance metrics and statistics
        """
        recent_trades = self.trade_history[-10:] if len(self.trade_history) >= 10 else self.trade_history
        recent_success_count = sum(1 for t in recent_trades if t.status == ExecutionStatus.EXECUTED)
        recent_success_rate = recent_success_count / len(recent_trades) if recent_trades else 0.0

        return {
            "total_trades": self.metrics.total_trades,
            "successful_trades": self.metrics.successful_trades,
            "failed_trades": self.metrics.failed_trades,
            "success_rate": self.metrics.success_rate,
            "recent_success_rate": recent_success_rate,
            "total_volume": self.metrics.total_volume,
            "average_execution_time": self.metrics.average_execution_time,
            "active": self.is_active,
            "last_trade": (
                self.trade_history[-1].signal.timestamp
                if self.trade_history else None
            )
        }

    def __enter__(self):
        """Context manager entry"""
        self.start_execution()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_execution()

        if exc_type:
            self.logger.error(
                "TradeExecutor context exit with exception",
                exception_type=exc_type.__name__ if exc_type else None,
                exception_message=str(exc_val) if exc_val else None
            )
