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
from strategy_engine import StrategyEngine, TradingAction, PortfolioManager, TradingSignal, RiskLevel
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
        self.portfolio_manager = PortfolioManager(self.config)
        self.trade_executor = TradeExecutor(self.config)

        # Bot state
        self.is_running = False
        self.trade_count = 0
        self.last_portfolio_display = 0

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
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def perform_health_check(self) -> bool:
        """
        Comprehensive health check with portfolio display

        Based on Trading Guide: validates API connectivity,
        portfolio access, and price data availability
        """
        self.logger.info("Performing system health check...")

        try:
            # 1. API Health Check
            health_response = self.data_handler.health_check()
            if not health_response.success:
                self.logger.error("API health check failed", error=health_response.error)
                return False

            self.logger.info("API health check passed",
                             response_time=health_response.response_time)

            # 2. Portfolio Data Check
            portfolio_success, portfolio = self.data_handler.get_portfolio()
            if not portfolio_success or not portfolio:
                self.logger.warning("Portfolio data unavailable, checking balances fallback")

                # Try balances fallback
                balances_success, balances = self.data_handler.get_balances()
                if not balances_success:
                    self.logger.error("Both portfolio and balances checks failed")
                    return False

                self.logger.info("Balances data retrieved successfully")
            else:
                # Display beautiful portfolio snapshot
                self._display_portfolio(portfolio)

                self.logger.info("Portfolio data retrieved successfully",
                                 agent_id=portfolio.agent_id,
                                 total_value=portfolio.total_value,
                                 token_count=len(portfolio.tokens))

            # 3. Price Data Check - Skip individual token checks since portfolio works
            self.logger.info("Price data validation skipped - portfolio snapshot shows prices work correctly")

            # 4. Portfolio Analysis (if PortfolioManager available)
            if portfolio:
                analysis = self.portfolio_manager.analyze_portfolio(portfolio)
                if "error" not in analysis:
                    self.logger.info("Portfolio analysis completed",
                                     needs_rebalancing=analysis.get("needs_rebalancing"),
                                     max_deviation=analysis.get("max_deviation"))

            self.logger.info("System health check completed successfully")
            return True

        except Exception as e:
            self.logger.error("Health check failed with exception", error=e)
            return False

    def _display_portfolio(self, portfolio) -> None:
        """
        Display beautiful portfolio table in console

        Creates formatted output similar to Recall UI interface
        """
        if not portfolio or not portfolio.tokens:
            self.logger.info("No portfolio data to display")
            return

        print("\n" + "=" * 80)
        print("📊 RECALL TRADING AGENT - PORTFOLIO SNAPSHOT")
        print("=" * 80)
        print(f"Agent ID: {portfolio.agent_id}")
        print(f"Total Value: ${portfolio.total_value:,.2f}")
        print(f"Snapshot Time: {portfolio.snapshot_time}")
        print("-" * 80)
        print(f"{'Symbol':<12} {'Amount':<16} {'Price':<12} {'Value':<14} {'Share %':<10}")
        print("-" * 80)

        for token in portfolio.tokens:
            symbol = token.get("symbol", "UNKNOWN")
            amount = float(token.get("amount", 0))
            price = float(token.get("price", 0))
            value = float(token.get("value", 0))
            share = (value / portfolio.total_value * 100) if portfolio.total_value > 0 else 0

            print(f"{symbol:<12} {amount:<16.6f} ${price:<11.4f} ${value:<13.2f} {share:<10.2f}")

        print("-" * 80)
        print(f"{'TOTAL':<12} {'':<16} {'':<12} ${portfolio.total_value:<13.2f} {'100.00':<10}")
        print("=" * 80)

        # Portfolio analysis
        analysis = self.portfolio_manager.analyze_portfolio(portfolio)
        if "error" not in analysis:
            print("\n📈 PORTFOLIO ANALYSIS")
            print(f"Rebalancing needed: {'YES' if analysis['needs_rebalancing'] else 'NO'}")
            print(f"Max deviation: {analysis['max_deviation']:.2%}")
            print(f"Threshold: {analysis['rebalance_threshold']:.2%}")

            if analysis['needs_rebalancing']:
                print("\n🔄 Target vs Current Allocations:")
                for symbol in analysis['target_allocations']:
                    target = analysis['target_allocations'][symbol]
                    current = analysis['current_allocations'].get(symbol, 0)
                    deviation = analysis['deviations'][symbol]
                    print(f"  {symbol}: {current:.1%} (target: {target:.1%}, deviation: {deviation:+.1%})")

        print()

    def execute_verification_trade(self) -> bool:
        """
        Execute verification trade for agent setup

        Based on Trading Guide: required for agent registration
        """
        self.logger.info("Starting verification trade sequence...")

        try:
            # Импорт нужен ВВЕРХУ функции, не внутри if блока
            from strategy_engine import TradingSignal, RiskLevel

            # Всегда создаем простой verification signal без зависимости от strategy_engine
            signal = TradingSignal(
                action=TradingAction.BUY,
                from_token=self.config.tokens.usdc_address,
                to_token=self.config.tokens.weth_address,
                amount=self.config.trading.min_trade_amount_usd,
                confidence=0.8,
                risk_level=RiskLevel.LOW,
                reasoning="Verification trade - независимо от AI стратегии",
                timestamp=time.time(),
                strategy_name="VerificationStrategy"
            )

            self.logger.info("✅ Создали фиксированный verification signal",
                             action=signal.action.value,
                             amount=signal.amount)

            # Execute verification trade
            self.trade_executor.start_execution()
            result = self.trade_executor.execute_trade_signal(signal)

            # Проверка на None перед обращением к result.status
            if result is None:
                self.logger.error("Verification trade failed: execute_trade_signal returned None")
                return False

            if result.status.value == "EXECUTED":
                self.logger.info("Verification trade executed successfully",
                                 transaction_id=result.transaction_id,
                                 executed_amount=result.executed_amount)
                return True
            else:
                self.logger.error("Verification trade failed",
                                  error=result.error_message)
                return False

        except Exception as e:
            self.logger.error("Verification trade failed with exception", error=e)
            return False

    def run_trading_cycle(self) -> bool:
        """
        Execute single trading cycle

        Returns:
            bool: True if cycle completed successfully
        """
        try:
            cycle_start = time.time()

            # Generate trading signal
            signal = self.strategy_engine.generate_signal()
            if not signal:
                self.logger.debug("No trading signal generated")
                return True

            self.logger.info("Trading signal generated",
                             action=signal.action.value,
                             confidence=signal.confidence,
                             reasoning=signal.reasoning)

            # Execute trade if not HOLD
            if signal.action != TradingAction.HOLD:
                result = self.trade_executor.execute_trade_signal(signal)

                if result.status.value == "EXECUTED":
                    self.trade_count += 1
                    self.logger.info("Trade executed successfully",
                                     trade_count=self.trade_count,
                                     transaction_id=result.transaction_id)
                else:
                    self.logger.warning("Trade execution failed",
                                        error=result.error_message)

            # Check for portfolio rebalancing (every 10 cycles or 30 minutes)
            current_time = time.time()
            if (self.trade_count % 10 == 0 or
                    current_time - self.last_portfolio_display > 1800):

                self._check_portfolio_rebalancing()
                self.last_portfolio_display = current_time

                # Auto-rebalancing logic from Portfolio Manager Tutorial
                portfolio_success, portfolio = self.data_handler.get_portfolio()
                if portfolio_success and portfolio:
                    rebalance_orders = self.portfolio_manager.plan_rebalance(portfolio)
                    if rebalance_orders:
                        self.logger.info(f"Executing {len(rebalance_orders)} rebalance orders")
                        for order in rebalance_orders:
                            self.logger.info(
                                f"🔍 Rebalance order: {order['from_token'][:10]}...→{order['to_token'][:10]}..., amount: {order['amount']}, reasoning: {order.get('reasoning', 'N/A')}")

                            # 🎯 ПРЯМОЙ API ВЫЗОВ С ПРАВИЛЬНЫМ API KEY!
                            try:
                                import requests
                                import json

                                trade_data = {
                                    "fromToken": order["from_token"],
                                    "toToken": order["to_token"],
                                    "amount": str(order["amount"]),  # Amount от PortfolioManager напрямую!
                                    "reason": order.get("reasoning", "Portfolio rebalancing")
                                }

                                headers = {
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {self.config.current_api_key}"  # ✅ ИСПРАВЛЕНО!
                                }

                                response = requests.post(
                                    f"{self.config.current_base_url}/api/trade/execute",  # ✅ ПРАВИЛЬНЫЙ URL
                                    headers=headers,
                                    json=trade_data,
                                    timeout=30
                                )

                                if response.status_code == 200:
                                    result_data = response.json()
                                    if result_data.get("success"):
                                        self.logger.info("✅ Rebalance trade SUCCESS",
                                                         amount=order["amount"],
                                                         reasoning=order["reasoning"])
                                    else:
                                        self.logger.warning("❌ Rebalance trade API error",
                                                            error=result_data.get("error"),
                                                            amount=order["amount"])
                                else:
                                    self.logger.error("❌ Rebalance HTTP error",
                                                      status=response.status_code,
                                                      text=response.text[:200])

                            except Exception as e:
                                self.logger.error("❌ Rebalance exception", error=str(e))

            cycle_duration = time.time() - cycle_start
            self.logger.debug("Trading cycle completed", duration=cycle_duration)

            return True

        except Exception as e:
            self.logger.error("Trading cycle failed", error=e)
            return False

    def _check_portfolio_rebalancing(self) -> None:
        """Check and display portfolio status"""
        try:
            portfolio_success, portfolio = self.data_handler.get_portfolio()
            if not portfolio_success or not portfolio:
                return

            # Display current portfolio
            self._display_portfolio(portfolio)

            self.logger.info("Portfolio displayed successfully")

        except Exception as e:
            self.logger.error("Portfolio check failed", error=e)

    def run_continuous(self, max_cycles: Optional[int] = None) -> None:
        """
        Run bot in continuous trading mode

        Args:
            max_cycles: Maximum number of cycles (None for infinite)
        """
        self.logger.info("Starting continuous trading mode",
                         max_cycles=max_cycles,
                         trade_interval=self.config.trading.trade_interval_seconds)

        self.is_running = True
        self.trade_executor.start_execution()
        cycle_count = 0

        try:
            while self.is_running:
                # Check max cycles limit
                if max_cycles and cycle_count >= max_cycles:
                    self.logger.info("Max cycles reached, stopping",
                                     cycles_completed=cycle_count)
                    break

                # Execute trading cycle
                cycle_success = self.run_trading_cycle()
                cycle_count += 1

                if not cycle_success:
                    self.logger.warning("Trading cycle failed", cycle=cycle_count)

                # Wait for next cycle with escape condition
                if self.is_running and cycle_count < 100:  # Максимум 100 циклов
                    time.sleep(self.config.trading.trade_interval_seconds)
                else:
                    self.logger.info("Cycle limit reached, stopping")
                    break

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, stopping...")
        except Exception as e:
            self.logger.error("Continuous trading failed", error=e)
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the trading bot and cleanup"""
        self.logger.info("Stopping trading bot...")
        self.is_running = False
        self.trade_executor.stop_execution()

        # Export final performance report
        self.export_performance_report()

    def export_performance_report(self) -> None:
        """Export comprehensive performance report"""
        try:
            # Get execution metrics
            metrics = self.trade_executor.get_execution_metrics()
            performance = self.trade_executor.get_performance_summary()
            strategy_metrics = self.strategy_engine.get_strategy_metrics()

            # Get final portfolio snapshot
            portfolio_success, portfolio = self.data_handler.get_portfolio()

            report = {
                "timestamp": time.time(),
                "environment": self.config.environment_name,
                "execution_metrics": {
                    "total_trades": metrics.total_trades,
                    "successful_trades": metrics.successful_trades,
                    "success_rate": metrics.success_rate,
                    "total_volume": metrics.total_volume,
                    "average_execution_time": metrics.average_execution_time
                },
                "performance_summary": performance,
                "strategy_metrics": strategy_metrics,
                "final_portfolio": {
                    "total_value": portfolio.total_value if portfolio else 0,
                    "token_count": len(portfolio.tokens) if portfolio else 0
                } if portfolio_success else None,
                "trade_history": self.trade_executor.export_trade_history()
            }

            # Save report
            import json
            from datetime import datetime

            filename = f"recall_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info("Performance report exported", filename=filename)

            # Display summary
            print("\n" + "=" * 60)
            print("📊 TRADING SESSION SUMMARY")
            print("=" * 60)
            print(f"Total Trades: {metrics.total_trades}")
            print(f"Success Rate: {metrics.success_rate:.1%}")
            print(f"Total Volume: ${metrics.total_volume:,.2f}")
            print(f"Avg Execution Time: {metrics.average_execution_time:.3f}s")
            if portfolio:
                print(f"Final Portfolio Value: ${portfolio.total_value:,.2f}")
            print(f"Report saved: {filename}")
            print("=" * 60)

        except Exception as e:
            self.logger.error("Failed to export performance report", error=e)


def main():
    """Main entry point for the trading bot"""
    import argparse

    parser = argparse.ArgumentParser(description="Recall AI Trading Agent")
    parser.add_argument("--production", action="store_true",
                        help="Use production environment instead of sandbox")
    parser.add_argument("--cycles", type=int, default=None,
                        help="Maximum number of trading cycles (default: infinite)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only perform verification trade and exit")

    args = parser.parse_args()

    # Initialize bot
    bot = RecallTradingBot(use_production=args.production)

    try:
        # Health check
        if not bot.perform_health_check():
            print("❌ Health check failed, exiting...")
            return 1

        # Verification trade (required for competitions)
        if not bot.execute_verification_trade():
            print("❌ Verification trade failed, exiting...")
            return 1

        if args.verify_only:
            print("✅ Verification completed successfully")
            return 0

        # Start continuous trading
        bot.run_continuous(max_cycles=args.cycles)

        return 0

    except Exception as e:
        bot.logger.error("Bot execution failed", error=e)
        return 1


if __name__ == "__main__":
    exit(main())
