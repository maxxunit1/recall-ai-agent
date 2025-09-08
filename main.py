"""
Main Entry Point

Enterprise trading bot orchestrator that coordinates all modules
for automated trading execution with comprehensive monitoring.
"""

import sys
import time
import signal
import asyncio
from typing import Optional
from pathlib import Path

# Add current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config
from utils import Logger
from data_handler import DataHandler
from strategy_engine import StrategyEngine, TradingAction
from executor import TradeExecutor


class RecallTradingBot:
    """
    Main trading bot orchestrator

    Coordinates all modules for automated trading execution with
    comprehensive error handling and performance monitoring.
    """

    def __init__(self, use_production: bool = False):
        """
        Initialize trading bot with all components

        Args:
            use_production: Whether to use production environment
        """
        # Override production setting in config
        import os
        os.environ["RECALL_USE_PRODUCTION"] = str(use_production)

        # Initialize core components
        self.config = Config()
        self.logger = Logger.get_logger("RecallTradingBot")
        self.data_handler = DataHandler(self.config)
        self.strategy_engine = StrategyEngine(self.config)
        self.executor = TradeExecutor(self.config)

        # Bot state
        self.is_running = False
        self.trade_count = 0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info(
            "RecallTradingBot initialized",
            environment=self.config.environment_name,
            config=self.config.to_dict()
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("Shutdown signal received", signal=signum)
        self.stop()

    def perform_health_check(self) -> bool:
        """
        Perform comprehensive system health check

        Returns:
            bool: True if all systems are healthy
        """
        self.logger.info("Performing system health check...")

        try:
            # Test API connectivity
            health_response = self.data_handler.health_check()
            if not health_response.success:
                self.logger.error("API health check failed", error=health_response.error)
                return False

            self.logger.info("API health check passed", response_time=health_response.response_time)

            # Test data retrieval
            portfolio_success, portfolio = self.data_handler.get_portfolio()
            if portfolio_success and portfolio:
                self.logger.info(
                    "Portfolio data retrieved successfully",
                    agent_id=portfolio.agent_id,
                    total_value=portfolio.total_value,
                    token_count=len(portfolio.tokens)
                )
            else:
                self.logger.warning("Portfolio data retrieval failed")

            # Test token price retrieval
            price_success, price_data = self.data_handler.get_token_price(
                self.config.tokens.usdc_address
            )
            if price_success and price_data:
                self.logger.info("Token price retrieval successful", price=price_data.price)
            else:
                self.logger.warning("Token price retrieval failed")

            self.logger.info("System health check completed successfully")
            return True

        except Exception as e:
            self.logger.error("Health check failed with exception", error=e)
            return False

    def execute_verification_trade(self) -> bool:
        """
        Execute verification trade for agent setup

        Returns:
            bool: True if verification successful
        """
        self.logger.info("Starting verification trade sequence...")

        try:
            # Generate verification signal
            signal = self.strategy_engine.generate_signal()
            if not signal:
                self.logger.error("Failed to generate verification signal")
                return False

            # Force a trade action for verification (override HOLD)
            if signal.action == TradingAction.HOLD:
                # Create a simple verification trade
                from strategy_engine import TradingSignal, RiskLevel

                signal = TradingSignal(
                    action=TradingAction.BUY,
                    from_token=self.config.tokens.usdc_address,
                    to_token=self.config.tokens.weth_address,
                    amount=self.config.trading.default_amount,
                    confidence=0.8,
                    risk_level=RiskLevel.LOW,
                    reasoning="Verification trade for agent setup",
                    timestamp=time.time(),
                    strategy_name="VerificationStrategy"
                )

            # Execute verification trade
            self.executor.start_execution()
            result = self.executor.execute_trade(signal)

            if result.status.value == "EXECUTED":
                self.logger.info(
                    "Verification trade executed successfully",
                    transaction_id=result.transaction_id,
                    executed_amount=result.executed_amount
                )
                return True
            else:
                self.logger.error(
                    "Verification trade failed",
                    status=result.status.value,
                    error=result.error_message
                )
                return False

        except Exception as e:
            self.logger.error("Verification trade failed with exception", error=e)
            return False

    def run_single_cycle(self) -> bool:
        """
        Run single trading cycle

        Returns:
            bool: True if cycle completed successfully
        """
        try:
            self.logger.debug("Starting trading cycle", cycle_count=self.trade_count + 1)

            # Generate trading signal
            signal = self.strategy_engine.generate_signal()
            if not signal:
                self.logger.debug("No trading signal generated")
                return True  # No signal is not an error

            self.logger.info(
                "Trading signal generated",
                action=signal.action.value,
                confidence=signal.confidence,
                strategy=signal.strategy_name
            )

            # Execute trade if action required
            if signal.action != TradingAction.HOLD:
                result = self.executor.execute_trade(signal)

                if result.status.value == "EXECUTED":
                    self.trade_count += 1
                    self.logger.info(
                        "Trade executed successfully",
                        trade_count=self.trade_count,
                        transaction_id=result.transaction_id
                    )
                else:
                    self.logger.warning(
                        "Trade execution failed",
                        error=result.error_message
                    )
            else:
                self.logger.info("Signal indicates HOLD - no trade executed")

            return True

        except Exception as e:
            self.logger.error("Trading cycle failed", error=e)
            return False

    def run_continuous(self, max_cycles: Optional[int] = None) -> None:
        """
        Run continuous trading with specified number of cycles

        Args:
            max_cycles: Maximum number of trading cycles (None for infinite)
        """
        self.logger.info(
            "Starting continuous trading",
            max_cycles=max_cycles,
            trade_interval=self.config.trading.trade_interval_seconds
        )

        self.is_running = True
        self.executor.start_execution()
        cycles_completed = 0

        try:
            while self.is_running:
                if max_cycles and cycles_completed >= max_cycles:
                    self.logger.info("Maximum cycles reached", cycles=cycles_completed)
                    break

                # Execute trading cycle
                cycle_success = self.run_single_cycle()
                cycles_completed += 1

                if not cycle_success:
                    self.logger.warning("Trading cycle failed, continuing...")

                # Log periodic status
                if cycles_completed % 10 == 0:
                    metrics = self.executor.get_execution_metrics()
                    self.logger.info(
                        "Periodic status update",
                        cycles_completed=cycles_completed,
                        total_trades=metrics.total_trades,
                        success_rate=metrics.success_rate
                    )

                # Wait before next cycle
                if self.is_running:
                    time.sleep(self.config.trading.trade_interval_seconds)

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error("Continuous trading failed", error=e)
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop trading bot gracefully"""
        if not self.is_running:
            return

        self.logger.info("Stopping trading bot...")
        self.is_running = False
        self.executor.stop_execution()

        # Cancel any pending trades
        cancelled_count = self.executor.cancel_pending_trades()
        if cancelled_count > 0:
            self.logger.info("Cancelled pending trades", count=cancelled_count)

        # Log final metrics
        metrics = self.executor.get_execution_metrics()
        strategy_metrics = self.strategy_engine.get_strategy_metrics()

        self.logger.info(
            "Trading bot stopped - Final metrics",
            total_cycles=self.trade_count,
            execution_metrics=metrics.__dict__,
            strategy_metrics=strategy_metrics
        )

    def export_performance_report(self) -> dict:
        """
        Export comprehensive performance report

        Returns:
            dict: Performance report data
        """
        execution_metrics = self.executor.get_execution_metrics()
        strategy_metrics = self.strategy_engine.get_strategy_metrics()
        trade_history = self.executor.export_trade_history()

        return {
            "bot_info": {
                "environment": self.config.environment_name,
                "total_cycles": self.trade_count,
                "is_running": self.is_running
            },
            "execution_metrics": execution_metrics.__dict__,
            "strategy_metrics": strategy_metrics,
            "trade_history": trade_history,
            "config_summary": self.config.to_dict()
        }


def main():
    """Main entry point for the trading bot"""
    print("=" * 60)
    print("🤖 RECALL AI TRADING AGENT v2.0")
    print("Enterprise-Grade Modular Architecture")
    print("=" * 60)

    try:
        # Initialize bot (default to sandbox)
        bot = RecallTradingBot(use_production=False)

        # Perform health check
        print("\n📋 STEP 1: System Health Check")
        if not bot.perform_health_check():
            print("❌ Health check failed. Exiting.")
            return 1
        print("✅ Health check passed")

        # Execute verification trade
        print("\n🔄 STEP 2: Agent Verification")
        if not bot.execute_verification_trade():
            print("❌ Verification failed. Exiting.")
            return 1
        print("✅ Agent verification successful")

        print("\n🎯 GOAL ACHIEVED: Agent registered and verified in Recall Network!")
        print("\n📈 Next Steps:")
        print("   1. Review competition calendar: https://docs.recall.network/competitions")
        print("   2. Enhance trading strategies")
        print("   3. Participate in live competitions")

        # Optional: Run a few trading cycles for demonstration
        print("\n🔄 STEP 3: Demo Trading Cycles (Optional)")
        user_choice = input("Run demo trading cycles? (y/N): ").lower().strip()

        if user_choice == 'y':
            print("Running 3 demo trading cycles...")
            bot.run_continuous(max_cycles=3)

            # Export performance report
            report = bot.export_performance_report()
            print(f"\n📊 Performance Summary:")
            print(f"   - Total Trades: {report['execution_metrics']['total_trades']}")
            print(f"   - Success Rate: {report['execution_metrics']['success_rate']:.1%}")
            print(f"   - Environment: {report['bot_info']['environment']}")

        print("\n🏁 Trading bot demonstration completed successfully!")
        return 0

    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
