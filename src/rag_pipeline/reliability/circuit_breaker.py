"""Circuit breaker — prevent cascading failures from external services."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

        try:
            result = breaker.call(risky_function, arg1, arg2)
        except CircuitBreakerOpenError:
            # Service is down, use fallback
            result = fallback_value
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        name: str = "default",
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for recovery."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker %s: OPEN → HALF_OPEN", self._name)
        return self._state

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute function through circuit breaker."""
        current_state = self.state

        if current_state == CircuitState.OPEN:
            logger.warning(
                "Circuit breaker %s: rejecting call (OPEN)", self._name
            )
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self._name}' is OPEN. "
                f"Service unavailable for {self._recovery_timeout}s."
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= 2:
                logger.info("Circuit breaker %s: HALF_OPEN → CLOSED", self._name)
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        else:
            self._failure_count = 0

    def _on_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self._failure_threshold:
            if self._state != CircuitState.OPEN:
                logger.warning(
                    "Circuit breaker %s: %d failures → OPEN",
                    self._name,
                    self._failure_count,
                )
            self._state = CircuitState.OPEN
            self._success_count = 0

    def reset(self) -> None:
        """Reset circuit breaker to CLOSED."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    def get_stats(self) -> dict:
        """Return circuit breaker statistics."""
        return {
            "name": self._name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout": self._recovery_timeout,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
