"""
Data Handler Module

Enterprise-grade API client for Recall Network with comprehensive
error handling, retry logic, and performance monitoring.
"""

import json
import requests
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timezone

from config import Config
from utils import Logger, PerformanceTracker, retry_with_backoff, validate_api_response, sanitize_for_logging


@dataclass
class ApiResponse:
    """Structured API response container"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None


@dataclass
class PortfolioData:
    """Portfolio information structure"""
    agent_id: str
    total_value: float
    tokens: List[Dict[str, Any]]
    snapshot_time: str


@dataclass
class TokenPrice:
    """Token price information"""
    address: str
    price: float
    chain: str
    specific_chain: str


class DataHandler:
    """
    Enterprise data handler for Recall API interactions

    Provides comprehensive API client functionality with:
    - Structured error handling
    - Performance monitoring
    - Automatic retries
    - Response validation
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = Logger.get_logger("DataHandler")
        self.performance_tracker = PerformanceTracker(self.logger)

        # Setup session with default headers
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.current_api_key}",
            "User-Agent": "RecallAI-TradingAgent/2.0"
        })

        self.logger.info(
            "DataHandler initialized",
            environment=self.config.environment_name,
            base_url=self.config.current_base_url
        )

    def _make_request(
            self,
            method: str,
            endpoint: str,
            data: Optional[Dict[str, Any]] = None,
            params: Optional[Dict[str, Any]] = None,
            operation_id: Optional[str] = None
    ) -> ApiResponse:
        """
        Make HTTP request with comprehensive error handling

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request payload for POST/PUT
            params: URL parameters for GET
            operation_id: Operation identifier for tracking

        Returns:
            ApiResponse: Structured response with success/error info
        """
        operation_id = operation_id or f"{method}_{endpoint.replace('/', '_')}"
        url = f"{self.config.current_base_url}{endpoint}"

        self.performance_tracker.start_operation(operation_id)

        try:
            # Prepare request arguments
            request_kwargs = {
                "timeout": self.config.api.timeout,
                "headers": self.session.headers
            }

            if data:
                request_kwargs["json"] = data
            if params:
                request_kwargs["params"] = params

            # Make request
            response = requests.request(method, url, **request_kwargs)
            duration = self.performance_tracker.end_operation(operation_id, success=True)

            # Handle different response scenarios
            if response.status_code == 200:
                try:
                    response_data = response.json() if response.content else {}

                    # Handle successful responses
                    if validate_api_response(response_data):
                        self.logger.info("API request successful",
                                       operation_id=operation_id,
                                       status_code=response.status_code)
                        return ApiResponse(
                            success=True,
                            data=response_data,
                            status_code=response.status_code,
                            response_time=duration
                        )
                    else:
                        # Response format validation failed
                        self.logger.error("Invalid API response format",
                                        operation_id=operation_id,
                                        response_data=sanitize_for_logging(response_data))
                        return ApiResponse(
                            success=False,
                            error="Invalid response format",
                            status_code=response.status_code,
                            response_time=duration
                        )

                except json.JSONDecodeError as e:
                    self.logger.error("JSON decode error",
                                    operation_id=operation_id,
                                    error=str(e))
                    return ApiResponse(
                        success=False,
                        error=f"JSON decode error: {str(e)}",
                        status_code=response.status_code,
                        response_time=duration
                    )
            else:
                # Handle HTTP errors
                error_message = response.text if response.content else f"HTTP {response.status_code}"

                # Add hint for 5xx errors (RPC issues from Discord chat)
                if 500 <= response.status_code < 600:
                    error_message += " | Hint: Possible RPC/server issue, retry may help"

                self.logger.error("API request failed",
                                operation_id=operation_id,
                                status_code=response.status_code,
                                error_message=error_message)

                return ApiResponse(
                    success=False,
                    error=error_message,
                    status_code=response.status_code,
                    response_time=duration
                )

        except requests.exceptions.Timeout:
            self.performance_tracker.end_operation(operation_id, success=False)
            self.logger.error("Request timeout", operation_id=operation_id, url=url)
            return ApiResponse(success=False, error="Request timeout")

        except requests.exceptions.ConnectionError:
            self.performance_tracker.end_operation(operation_id, success=False)
            self.logger.error("Connection error", operation_id=operation_id, url=url)
            return ApiResponse(success=False, error="Connection error")

        except Exception as e:
            self.performance_tracker.end_operation(operation_id, success=False)
            self.logger.error("Unexpected error", operation_id=operation_id, error=e)
            return ApiResponse(success=False, error=f"Unexpected error: {str(e)}")

    @retry_with_backoff(max_retries=3, exceptions=(requests.exceptions.RequestException,))
    def health_check(self) -> ApiResponse:
        """
        Perform health check on API

        Returns:
            ApiResponse: Health check result
        """
        return self._make_request("GET", "/api/health", operation_id="health_check")

    def get_balances(self) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
        """
        Get agent balances (list format)

        Based on API Reference: GET /api/agent/balances
        Returns list of balance objects with token details
        """
        resp = self._make_request("GET", "/api/agent/balances", operation_id="get_balances")
        if not resp.success or not resp.data:
            return False, None

        data = resp.data.get("balances") if isinstance(resp.data, dict) else resp.data
        if isinstance(data, list):
            return True, data

        self.logger.warning("Unexpected balances response format", response=resp.data)
        return False, None

    def get_simple_balance(self) -> Tuple[bool, Optional[Dict[str, float]]]:
        """
        Get simple balance map format

        Based on Portfolio Manager Tutorial: GET /api/balance
        Returns {"USDC": 123.45, "WETH": 0.56, ...}
        """
        resp = self._make_request("GET", "/api/balance", operation_id="get_simple_balance")
        if not resp.success or not resp.data or not isinstance(resp.data, dict):
            return False, None
        return True, resp.data

    def build_portfolio_snapshot(self, balances, price=None) -> str:
        """
        Build formatted portfolio snapshot for logging

        Args:
            balances: List of balance dicts with amount, token/symbol, chain, specificChain
            price: Optional USDC price for stablecoin fallback

        Returns:
            Formatted string with portfolio table
        """
        rows = []
        total_value = 0.0

        price_cache = {}

        for b in balances:
            token_addr = b.get("token") or b.get("tokenAddress") or b.get("address")
            amount = float(b.get("amount", 0))
            chain = b.get("chain", "evm")
            specific_chain = b.get("specificChain", "eth")
            symbol = b.get("symbol") or "UNKNOWN"

            # Get price for this token with caching
            price = 0.0
            if token_addr:
                cache_key = f"{token_addr}_{chain}_{specific_chain}"
                if cache_key not in price_cache:
                    ok_price, price_val = self.get_token_price(token_address=token_addr, chain=chain,
                                                               specific_chain=specific_chain)
                    if ok_price and price_val is not None:
                        price_cache[cache_key] = float(price_val)
                    else:
                        price_cache[cache_key] = 0.0
                price = price_cache[cache_key]

            value = amount * price
            total_value += value

            tokens_enriched.append({
                "token": token_addr,
                "symbol": symbol,
                "amount": amount,
                "price": price,
                "value": value,
                "chain": chain,
                "specificChain": specific_chain,
            })

        # Format output like in logs
        lines = []
        lines.append("\n📦 Portfolio Snapshot")
        lines.append(f"Snapshot: {datetime.now(timezone.utc).isoformat()}")
        lines.append("Symbol             Amount          Price            Value   Share%")
        lines.append("-------------------------------------------------------------------")

        for r in rows:
            share = (r["value"] / total_value * 100.0) if total_value > 0 else 0.0
            lines.append(
                f"{r['symbol']:<15} {r['amount']:>14.6f} {r['price']:>14.6f} {r['value']:>14.2f} {share:>8.2f}"
            )

        lines.append("--------------------------------------------------------------------")
        lines.append(f"Total Value: {total_value:.2f}")

        return "\n".join(lines)

    def get_portfolio(self) -> Tuple[bool, Optional[PortfolioData]]:
        """
        Universal portfolio reading with multiple fallbacks

        Based on Trading Guide: tries multiple endpoints:
        1) /api/agent/portfolio (Trading guide)
        2) /api/agent/balances (API Reference)
        3) /api/balance (Portfolio Manager tutorial)
        """
        # 1) Trading guide path
        response = self._make_request("GET", "/api/agent/portfolio", operation_id="get_portfolio")
        if response.success and response.data:
            try:
                portfolio = PortfolioData(
                    agent_id=response.data.get("agentId", "unknown"),
                    total_value=float(response.data.get("totalValue", 0.0)),
                    tokens=response.data.get("tokens", []),
                    snapshot_time=response.data.get("snapshotTime", datetime.now(timezone.utc).isoformat())
                )
                return True, portfolio
            except (KeyError, TypeError, ValueError) as e:
                self.logger.error("Invalid portfolio response structure", missing_field=str(e))
                # Continue to fallback

        # 2) Fallback: /api/agent/balances -> build portfolio with prices
        if response.status_code == 404:  # Expected when endpoint not available
            ok_bals, balances = self.get_balances()
            if ok_bals and balances:
                tokens_enriched = []
                total_value = 0.0
                price_cache = {}

                for b in balances:
                    token_addr = b.get("token") or b.get("tokenAddress") or b.get("address")
                    amount = float(b.get("amount", 0))
                    chain = b.get("chain", "evm")
                    specific_chain = b.get("specificChain", "eth")
                    symbol = b.get("symbol") or "UNKNOWN"

                    # Get price for this token with caching
                    price = 0.0
                    if token_addr:
                        cache_key = f"{token_addr}_{chain}_{specific_chain}"
                        if cache_key not in price_cache:
                            ok_price, price_val = self.get_token_price(token_address=token_addr, chain=chain,
                                                                       specific_chain=specific_chain)
                            if ok_price and price_val is not None:
                                price_cache[cache_key] = float(price_val)
                            else:
                                price_cache[cache_key] = 0.0
                        price = price_cache[cache_key]

                    value = amount * price
                    total_value += value

                    tokens_enriched.append({
                        "token": token_addr,
                        "symbol": symbol,
                        "amount": amount,
                        "price": price,
                        "value": value,
                        "chain": chain,
                        "specificChain": specific_chain,
                    })

                portfolio = PortfolioData(
                    agent_id="unknown",
                    total_value=total_value,
                    tokens=tokens_enriched,
                    snapshot_time=datetime.now(timezone.utc).isoformat()
                )
                self.logger.info("Portfolio built from balances+price",
                               total_value=total_value,
                               tokens=len(tokens_enriched))
                return True, portfolio

        # 3) Fallback: /api/balance (Portfolio Manager tutorial) -> convert to tokens[] format
        ok, simple = self.get_simple_balance()
        if ok and simple:
            tokens_enriched = []
            total_value = 0.0
            for symbol, amount in simple.items():
                amount = float(amount)
                price = 0.0  # Would need separate price lookup
                value = 0.0
                total_value += value
                tokens_enriched.append({
                    "token": None,
                    "symbol": symbol,
                    "amount": amount,
                    "price": price,
                    "value": value,
                    "chain": "evm",
                    "specificChain": "eth",
                })

            portfolio = PortfolioData(
                agent_id="unknown",
                total_value=total_value,
                tokens=tokens_enriched,
                snapshot_time=datetime.now(timezone.utc).isoformat()
            )
            self.logger.info("Portfolio built from /api/balance (symbol map)")
            return True, portfolio

        return False, None

    def get_token_price(
            self,
            token_address: Optional[str] = None,
            chain: str = "evm",
            specific_chain: str = "eth",
    ) -> Tuple[bool, Optional[float]]:
        """
        Universal price fetching

        Based on API Reference: GET /api/price
        - without token_address -> default price (usually USDC)
        - with token_address -> specific token price with chain params
        """
        op_id = "get_token_price" if token_address is None else "get_token_price_by_token"

        params = None
        if token_address:
            params = {"token": token_address, "chain": chain, "specificChain": specific_chain}

        resp = self._make_request("GET", "/api/price", params=params, operation_id=op_id)
        if not resp.success or resp.data is None:
            return False, None

        data = resp.data
        # Handle various response formats:
        # 1) Expected: {"price": 0.9996}
        # 2) Wrapped: {"data": {"price": ...}}
        # 3) Direct number/string
        price_val = None
        if isinstance(data, dict):
            if "price" in data:
                price_val = data["price"]
            elif "data" in data and isinstance(data["data"], dict) and "price" in data["data"]:
                price_val = data["data"]["price"]
        else:
            price_val = data  # number or string

        try:
            price_float = float(price_val)
            # USDC price normalization fix - if price > 1000, divide by 100000
            if price_float > 1000:
                price_float = price_float / 100000
                self.logger.info(f"Price normalized: {price_val} -> {price_float}")
            return True, price_float
        except (TypeError, ValueError):
            self.logger.error(
                "Unexpected price payload",
                payload=sanitize_for_logging(data),
                operation_id=op_id,
            )
            return False, None

    def get_leaderboard(self) -> ApiResponse:
        """
        Get competition leaderboard

        Based on Competition API: GET /api/competition/leaderboard
        """
        return self._make_request("GET", "/api/competition/leaderboard", operation_id="get_leaderboard")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        if hasattr(self, 'session'):
            self.session.close()

        if exc_type:
            self.logger.error(
                "DataHandler context exit with exception",
                exception_type=exc_type.__name__ if exc_type else None,
                exception_message=str(exc_val) if exc_val else None
            )