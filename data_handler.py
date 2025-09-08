"""
Data Handler Module

Enterprise-grade API client for Recall Network with comprehensive
error handling, retry logic, and performance monitoring.
"""

import json
import requests
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

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
            data: Request body data
            params: Query parameters
            operation_id: Unique operation identifier for tracking

        Returns:
            ApiResponse: Structured response object
        """
        if not operation_id:
            operation_id = f"{method}_{endpoint.replace('/', '_')}"

        self.performance_tracker.start_operation(operation_id)

        url = f"{self.config.current_base_url}/{endpoint.lstrip('/')}"

        try:
            self.logger.debug(
                "Making API request",
                method=method,
                url=url,
                params=params,
                data_keys=list(data.keys()) if data else None
            )

            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.config.api.timeout
            )

            duration = self.performance_tracker.end_operation(
                operation_id,
                success=response.ok,
                status_code=response.status_code,
                url=url
            )

            if response.ok:
                try:
                    response_data = response.json()

                    if validate_api_response(response_data):
                        self.logger.info(
                            "API request successful",
                            operation_id=operation_id,
                            status_code=response.status_code
                        )
                        return ApiResponse(
                            success=True,
                            data=response_data,
                            status_code=response.status_code,
                            response_time=duration
                        )
                    else:
                        self.logger.warning(
                            "Invalid API response structure",
                            operation_id=operation_id,
                            response_data=sanitize_for_logging(response_data)
                        )
                        return ApiResponse(
                            success=False,
                            error="Invalid response structure",
                            status_code=response.status_code,
                            response_time=duration
                        )

                except json.JSONDecodeError as e:
                    self.logger.error(
                        "Failed to decode JSON response",
                        operation_id=operation_id,
                        error=e
                    )
                    return ApiResponse(
                        success=False,
                        error=f"JSON decode error: {str(e)}",
                        status_code=response.status_code,
                        response_time=duration
                    )
            else:
                # Handle HTTP errors
                try:
                    error_data = response.json()
                    error_message = error_data.get("error", {}).get("message", response.text)
                except (json.JSONDecodeError, AttributeError):
                    error_message = response.text

                self.logger.error(
                    "API request failed",
                    operation_id=operation_id,
                    status_code=response.status_code,
                    error_message=error_message
                )

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
        # Health endpoint is always on production URL regardless of environment
        health_url = "https://api.competitions.recall.network"
        original_base_url = self.config.current_base_url

        # Temporarily override base URL for health check
        self.config._current_base_url_override = health_url

        try:
            return self._make_request("GET", "/api/health", operation_id="health_check")
        finally:
            # Restore original base URL
            delattr(self.config, '_current_base_url_override')

    def get_portfolio(self) -> Tuple[bool, Optional[PortfolioData]]:
        """
        Get agent portfolio information

        Returns:
            Tuple[bool, Optional[PortfolioData]]: Success status and portfolio data
        """
        response = self._make_request("GET", "/api/agent/portfolio", operation_id="get_portfolio")

        if response.success and response.data:
            try:
                portfolio = PortfolioData(
                    agent_id=response.data["agentId"],
                    total_value=response.data["totalValue"],
                    tokens=response.data["tokens"],
                    snapshot_time=response.data["snapshotTime"]
                )
                return True, portfolio
            except KeyError as e:
                self.logger.error("Invalid portfolio response structure", missing_field=str(e))
                return False, None

        return False, None

    def get_token_price(
            self,
            token_address: str,
            chain: str = "evm",
            specific_chain: str = "eth"
    ) -> Tuple[bool, Optional[TokenPrice]]:
        """
        Get token price information

        Args:
            token_address: Token contract address
            chain: Blockchain type (evm, svm)
            specific_chain: Specific chain (eth, polygon, base, solana)

        Returns:
            Tuple[bool, Optional[TokenPrice]]: Success status and price data
        """
        params = {
            "token": token_address,
            "chain": chain,
            "specificChain": specific_chain
        }

        response = self._make_request(
            "GET",
            "/api/price",
            params=params,
            operation_id="get_token_price"
        )

        if response.success and response.data:
            try:
                price_data = TokenPrice(
                    address=token_address,
                    price=response.data["price"],
                    chain=response.data["chain"],
                    specific_chain=response.data["specificChain"]
                )
                return True, price_data
            except KeyError as e:
                self.logger.error("Invalid price response structure", missing_field=str(e))
                return False, None

        return False, None

    def get_leaderboard(self) -> ApiResponse:
        """
        Get competition leaderboard

        Returns:
            ApiResponse: Leaderboard data
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
