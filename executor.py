"""
Trade Executor Module

Enterprise-grade trade execution engine with comprehensive
error handling, transaction tracking, and performance analytics.
"""

import time
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

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def execute_trade(self, signal: TradingSignal) -> TradeResult:
        """
        Execute trading signal with comprehensive error handling

        Args:
            signal: Trading signal to execute

        Returns:
            TradeResult: Execution result with detailed information
        """
        operation_id = f"execute_trade_{int(time.time())}"
        self.performance_tracker.start_operation(operation_id)

        self.logger.info(
            "Starting trade execution",
            action=signal.action.value,
            from_token=signal.from_token[:10] + "..." if signal.from_token else "N/A",
            to_token=signal.to_token[:10] + "..." if signal.to_token else "N/A",
            amount=signal.amount,
            strategy=signal.strategy_name
        )

        # Skip execution for HOLD signals
        if signal.action == TradingAction.HOLD:
            result = TradeResult(
                signal=signal,
                status=ExecutionStatus.EXECUTED,
                execution_time=0.0
            )
            self._update_metrics(result)
            return result

        try:
            # Pre-execution validation
            if not self._validate_execution_conditions(signal):
                result = TradeResult(
                    signal=signal,
                    status=ExecutionStatus.FAILED,
                    error_message="Pre-execution validation failed"
                )
                self._update_metrics(result)
                return result

            # Prepare trade payload
            trade_payload = self._prepare_trade_payload(signal)

            # Execute trade via API
            api_response = self._execute_api_trade(trade_payload, operation_id)

            # Process execution result
            result = self._process_execution_result(signal, api_response, operation_id)

            # Update metrics and history
            self._update_metrics(result)
            self.trade_history.append(result)

            execution_time = self.performance_tracker.end_operation(
                operation_id,
                success=(result.status == ExecutionStatus.EXECUTED)
            )
            result.execution_time = execution_time

            return result

        except Exception as e:
            self.performance_tracker.end_operation(operation_id, success=False)
            self.logger.error("Trade execution failed with exception", error=e)

            result = TradeResult(
                signal=signal,
                status=ExecutionStatus.FAILED,
                error_message=f"Execution exception: {str(e)}"
            )
            self._update_metrics(result)
            return result

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
            except ValueError:
                self.logger.error("Invalid signal: Amount is not a valid number")
                return False

        # Check confidence threshold
        if signal.confidence < 0.5:  # Minimum 50% confidence for execution
            self.logger.warning(
                "Trade execution skipped: Low confidence",
                confidence=signal.confidence
            )
            return False

        return True

    def _prepare_trade_payload(self, signal: TradingSignal) -> Dict[str, Any]:
        """
        Prepare API payload for trade execution

        Args:
            signal: Trading signal

        Returns:
            Dict[str, Any]: API payload
        """
        payload = {
            "fromToken": signal.from_token,
            "toToken": signal.to_token,
            "amount": signal.amount,
            "reason": f"{signal.strategy_name} strategy: {signal.reasoning}"
        }

        # Add optional parameters if available
        if hasattr(signal, 'from_chain') and signal.from_chain:
            payload["fromChain"] = signal.from_chain
        if hasattr(signal, 'to_chain') and signal.to_chain:
            payload["toChain"] = signal.to_chain

        self.logger.debug("Trade payload prepared", payload_keys=list(payload.keys()))
        return payload

    def _execute_api_trade(self, payload: Dict[str, Any], operation_id: str) -> ApiResponse:
        """
        Execute trade via Recall API

        Args:
            payload: Trade payload
            operation_id: Operation identifier

        Returns:
            ApiResponse: API response
        """
        self.logger.info("Executing trade via API")

        response = self.data_handler._make_request(
            method="POST",
            endpoint="/api/trade/execute",
            data=payload,
            operation_id=operation_id
        )

        return response

    def _process_execution_result(
            self,
            signal: TradingSignal,
            api_response: ApiResponse,
            operation_id: str
    ) -> TradeResult:
        """
        Process API response and create trade result

        Args:
            signal: Original trading signal
            api_response: API response
            operation_id: Operation identifier

        Returns:
            TradeResult: Processed trade result
        """
        if api_response.success and api_response.data:
            transaction_data = api_response.data.get("transaction", {})

            # Extract execution details
            transaction_id = transaction_data.get("id")
            executed_amount = transaction_data.get("fromAmount")
            executed_price = transaction_data.get("price")
            to_amount = transaction_data.get("toAmount")

            # Calculate actual return and slippage
            actual_return = None
            slippage = None

            if executed_amount and to_amount and executed_price:
                expected_to_amount = executed_amount * executed_price
                if expected_to_amount > 0:
                    slippage = abs(to_amount - expected_to_amount) / expected_to_amount

                # Simple return calculation (this would be more complex in real scenarios)
                if signal.expected_return:
                    actual_return = (to_amount - executed_amount) / executed_amount * 100

            self.logger.info(
                "Trade executed successfully",
                transaction_id=transaction_id,
                executed_amount=executed_amount,
                executed_price=executed_price
            )

            return TradeResult(
                signal=signal,
                status=ExecutionStatus.EXECUTED,
                transaction_id=transaction_id,
                executed_amount=executed_amount,
                executed_price=executed_price,
                actual_return=actual_return,
                slippage=slippage
            )

        else:
            error_message = api_response.error or "Unknown API error"
            self.logger.error("Trade execution failed", error=error_message)

            return TradeResult(
                signal=signal,
                status=ExecutionStatus.FAILED,
                error_message=error_message
            )

    def _update_metrics(self, result: TradeResult) -> None:
        """
        Update execution metrics with trade result

        Args:
            result: Trade result to process
        """
        self.metrics.total_trades += 1

        if result.status == ExecutionStatus.EXECUTED:
            self.metrics.successful_trades += 1
            if result.executed_amount:
                self.metrics.total_volume += result.executed_amount
        else:
            self.metrics.failed_trades += 1

        # Update success rate
        self.metrics.success_rate = (
                self.metrics.successful_trades / self.metrics.total_trades
        ) if self.metrics.total_trades > 0 else 0.0

        # Update average execution time
        if result.execution_time:
            total_time = self.metrics.average_execution_time * (self.metrics.total_trades - 1)
            self.metrics.average_execution_time = (
                                                          total_time + result.execution_time
                                                  ) / self.metrics.total_trades

    def start_execution(self) -> None:
        """Start trade execution engine"""
        self.is_active = True
        self.logger.info("Trade execution engine started")

    def stop_execution(self) -> None:
        """Stop trade execution engine"""
        self.is_active = False
        self.logger.info("Trade execution engine stopped")

    def get_execution_metrics(self) -> ExecutionMetrics:
        """Get current execution metrics"""
        return self.metrics

    def get_trade_history(self, limit: Optional[int] = None) -> List[TradeResult]:
        """
        Get trade execution history

        Args:
            limit: Optional limit on number of results

        Returns:
            List[TradeResult]: Trade history
        """
        if limit:
            return self.trade_history[-limit:]
        return self.trade_history.copy()

    def cancel_pending_trades(self) -> int:
        """
        Cancel any pending trades (placeholder for future implementation)

        Returns:
            int: Number of cancelled trades
        """
        cancelled_count = 0
        for result in self.trade_history:
            if result.status == ExecutionStatus.PENDING:
                result.status = ExecutionStatus.CANCELLED
                cancelled_count += 1

        if cancelled_count > 0:
            self.logger.info("Cancelled pending trades", count=cancelled_count)

        return cancelled_count

    def reset_metrics(self) -> None:
        """Reset execution metrics"""
        self.metrics = ExecutionMetrics()
        self.logger.info("Execution metrics reset")

    def export_trade_history(self) -> List[Dict[str, Any]]:
        """
        Export trade history as serializable data

        Returns:
            List[Dict[str, Any]]: Serializable trade history
        """
        return [
            {
                "timestamp": result.signal.timestamp,
                "action": result.signal.action.value,
                "status": result.status.value,
                "from_token": result.signal.from_token,
                "to_token": result.signal.to_token,
                "amount": result.signal.amount,
                "transaction_id": result.transaction_id,
                "executed_price": result.executed_price,
                "strategy": result.signal.strategy_name,
                "confidence": result.signal.confidence,
                "reasoning": result.signal.reasoning,
                "error_message": result.error_message
            }
            for result in self.trade_history
        ]
