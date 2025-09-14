"""
Utilities Module

Enterprise-grade logging, error handling, and helper functions
following 2025 senior development standards with structured logging.
"""

import json
import logging
import sys
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps
from pathlib import Path


class StructuredLogger:
    """
    Enterprise structured logger with context management

    Provides consistent logging format with contextual information,
    performance tracking, and error correlation.
    """

    def __init__(self, name: str, log_file: str = "recall_agent.log", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(log_file)

    def _setup_handlers(self, log_file: str) -> None:
        """Setup file and console handlers with structured formatting"""

        # Custom formatter for structured logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console handler with UTF-8 support
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _log_with_context(self, level: str, message: str, **context) -> None:
        """Log message with structured context"""
        log_entry = {
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **context
        }

        formatted_message = f"{message}"
        if context:
            formatted_message += f" | Context: {json.dumps(context, default=str)}"

        getattr(self.logger, level.lower())(formatted_message)

    def info(self, message: str, **context) -> None:
        """Log info message with context"""
        self._log_with_context("INFO", message, **context)

    def error(self, message: str, error: Optional[Exception] = None, **context) -> None:
        """Log error message with exception details"""
        if error:
            context.update({
                "error_type": type(error).__name__,
                "error_message": str(error)
            })
        self._log_with_context("ERROR", message, **context)

    def warning(self, message: str, **context) -> None:
        """Log warning message with context"""
        self._log_with_context("WARNING", message, **context)

    def debug(self, message: str, **context) -> None:
        """Log debug message with context"""
        self._log_with_context("DEBUG", message, **context)


class PerformanceTracker:
    """
    Performance tracking utility for monitoring API calls and operations
    """

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self._start_times: Dict[str, float] = {}

    def start_operation(self, operation_id: str) -> None:
        """Start tracking an operation"""
        self._start_times[operation_id] = time.time()
        self.logger.debug("Operation started", operation_id=operation_id)

    def end_operation(self, operation_id: str, success: bool = True, **context) -> float:
        """End tracking and log performance"""
        if operation_id not in self._start_times:
            self.logger.warning("Operation end called without start", operation_id=operation_id)
            return 0.0

        duration = time.time() - self._start_times.pop(operation_id)

        self.logger.info(
            "Operation completed",
            operation_id=operation_id,
            duration_seconds=round(duration, 3),
            success=success,
            **context
        )

        return duration


def retry_with_backoff(
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retry logic with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exceptions to catch and retry
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    delay = backoff_factor * (2 ** attempt)
                    await asyncio.sleep(delay)

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    delay = backoff_factor * (2 ** attempt)
                    time.sleep(delay)

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def validate_api_response(response_data: Dict[str, Any]) -> bool:
    """
    Validate API response structure

    Args:
        response_data: API response dictionary

    Returns:
        bool: True if response is valid
    """
    if not isinstance(response_data, dict):
        return False

    # Handle health endpoint format
    if "status" in response_data and response_data["status"] == "ok":
        return True

    # Handle trading endpoint format - success field is required
    if "success" not in response_data:
        return False

    # Both success=true and success=false are valid responses
    # success=false with error info is a valid API response
    return True


def sanitize_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize sensitive data for logging

    Args:
        data: Dictionary that may contain sensitive information

    Returns:
        Dict with sensitive fields masked
    """
    sensitive_fields = {
        "api_key", "authorization", "password", "secret", "token", "key"
    }

    sanitized = {}
    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in sensitive_fields):
            sanitized[key] = "*" * 8 + key[-4:] if len(str(value)) > 4 else "****"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_for_logging(value)
        else:
            sanitized[key] = value

    return sanitized


class Logger:
    """
    Main logger factory following singleton pattern
    """
    _instances: Dict[str, StructuredLogger] = {}

    @classmethod
    def get_logger(cls, name: str, **kwargs) -> StructuredLogger:
        """Get or create logger instance"""
        if name not in cls._instances:
            cls._instances[name] = StructuredLogger(name, **kwargs)
        return cls._instances[name]


# Export commonly used utilities
__all__ = [
    "Logger",
    "StructuredLogger",
    "PerformanceTracker",
    "retry_with_backoff",
    "validate_api_response",
    "sanitize_for_logging"
]
