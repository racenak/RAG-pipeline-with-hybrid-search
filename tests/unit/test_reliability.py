"""Tests for reliability patterns — circuit breaker and retry."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from rag_pipeline.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)
from rag_pipeline.reliability.retry import retry_with_backoff


# ------------------------------------------------------------------ #
#  CircuitBreaker
# ------------------------------------------------------------------ #


class TestCircuitBreaker:
    def test_starts_in_closed_state(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0, name="test")
        assert breaker.state == CircuitState.CLOSED

    def test_opens_after_failure_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0, name="test")
        failing_func = MagicMock(side_effect=RuntimeError("fail"))

        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

    def test_rejects_when_open(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0, name="test")
        failing_func = MagicMock(side_effect=RuntimeError("fail"))

        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(failing_func)

        # Now circuit is open — should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(lambda: "should not run")

    def test_transitions_to_half_open_after_recovery_timeout(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, name="test")
        failing_func = MagicMock(side_effect=RuntimeError("fail"))

        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

    def test_closes_after_successes_in_half_open(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, name="test")
        failing_func = MagicMock(side_effect=RuntimeError("fail"))

        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(failing_func)

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Two successes in half-open should close the circuit
        breaker.call(lambda: "ok")
        assert breaker.state == CircuitState.HALF_OPEN  # still half-open
        breaker.call(lambda: "ok")
        assert breaker.state == CircuitState.CLOSED

    def test_reset(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0, name="test")
        failing_func = MagicMock(side_effect=RuntimeError("fail"))

        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        assert breaker._success_count == 0

    def test_get_stats(self):
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0, name="test")
        stats = breaker.get_stats()

        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
        assert stats["success_count"] == 0
        assert stats["failure_threshold"] == 5
        assert stats["recovery_timeout"] == 30.0

    def test_success_resets_failure_count(self):
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=1.0, name="test")
        failing_func = MagicMock(side_effect=RuntimeError("fail"))

        # Two failures (below threshold)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(failing_func)

        # Success resets failure count
        breaker.call(lambda: "ok")
        assert breaker._failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    def test_failure_count_tracks_accurately(self):
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=1.0, name="test")
        failing_func = MagicMock(side_effect=RuntimeError("fail"))

        with pytest.raises(RuntimeError):
            breaker.call(failing_func)
        assert breaker._failure_count == 1

        with pytest.raises(RuntimeError):
            breaker.call(failing_func)
        assert breaker._failure_count == 2

    def test_call_returns_result(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0, name="test")
        result = breaker.call(lambda x: x * 2, 5)
        assert result == 10

    def test_call_passes_kwargs(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0, name="test")
        result = breaker.call(lambda a, b: a + b, a=3, b=4)
        assert result == 7

    def test_open_error_has_descriptive_message(self):
        breaker = CircuitBreaker(
            failure_threshold=1, recovery_timeout=10.0, name="my-service"
        )
        with pytest.raises(RuntimeError):
            breaker.call(MagicMock(side_effect=RuntimeError("fail")))

        with pytest.raises(CircuitBreakerOpenError, match="my-service"):
            breaker.call(lambda: "nope")


# ------------------------------------------------------------------ #
#  retry_with_backoff
# ------------------------------------------------------------------ #


class TestRetryWithBackoff:
    def test_succeeds_on_first_try(self):
        func = MagicMock(return_value=42)
        result = retry_with_backoff(func, max_retries=3)
        assert result == 42
        assert func.call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        func = MagicMock(side_effect=[ValueError("fail"), ValueError("fail"), "ok"])
        result = retry_with_backoff(func, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert func.call_count == 3

    def test_raises_after_max_retries_exhausted(self):
        func = MagicMock(side_effect=ValueError("always fail"))
        with pytest.raises(ValueError, match="always fail"):
            retry_with_backoff(func, max_retries=2, base_delay=0.01)
        assert func.call_count == 3  # 1 initial + 2 retries

    def test_calls_on_retry_callback(self):
        on_retry = MagicMock()
        func = MagicMock(side_effect=[ValueError("fail"), "ok"])
        retry_with_backoff(
            func, max_retries=3, base_delay=0.01, on_retry=on_retry
        )
        assert on_retry.call_count == 1
        call_args = on_retry.call_args[0]
        assert call_args[0] == 1  # attempt number
        assert isinstance(call_args[1], ValueError)

    def test_only_retries_specified_exceptions(self):
        func = MagicMock(side_effect=TypeError("wrong type"))
        with pytest.raises(TypeError):
            retry_with_backoff(
                func,
                max_retries=3,
                base_delay=0.01,
                retryable_exceptions=(ValueError,),
            )
        assert func.call_count == 1  # no retries for TypeError

    def test_retries_multiple_exception_types(self):
        func = MagicMock(
            side_effect=[ValueError("v"), TypeError("t"), "ok"]
        )
        result = retry_with_backoff(
            func,
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ValueError, TypeError),
        )
        assert result == "ok"
        assert func.call_count == 3

    def test_exponential_backoff_delays_increase(self):
        delays = []
        original_sleep = time.sleep

        def mock_sleep(delay):
            delays.append(delay)

        func = MagicMock(
            side_effect=[
                ValueError("1"),
                ValueError("2"),
                ValueError("3"),
                "ok",
            ]
        )

        import rag_pipeline.reliability.retry as retry_mod

        original_func = retry_mod.time.sleep
        retry_mod.time.sleep = mock_sleep
        try:
            retry_with_backoff(
                func, max_retries=3, base_delay=1.0, exponential=True, jitter=False
            )
        finally:
            retry_mod.time.sleep = original_func

        assert len(delays) == 3
        assert delays[0] == 1.0  # 2^0 * 1.0
        assert delays[1] == 2.0  # 2^1 * 1.0
        assert delays[2] == 4.0  # 2^2 * 1.0

    def test_linear_backoff_delays_increase(self):
        delays = []

        func = MagicMock(
            side_effect=[
                ValueError("1"),
                ValueError("2"),
                "ok",
            ]
        )

        import rag_pipeline.reliability.retry as retry_mod

        original_func = retry_mod.time.sleep
        retry_mod.time.sleep = lambda d: delays.append(d)
        try:
            retry_with_backoff(
                func,
                max_retries=3,
                base_delay=1.0,
                exponential=False,
                jitter=False,
            )
        finally:
            retry_mod.time.sleep = original_func

        assert len(delays) == 2
        assert delays[0] == 1.0  # 1 * (0+1)
        assert delays[1] == 2.0  # 1 * (1+1)

    def test_max_delay_caps_exponential(self):
        delays = []

        func = MagicMock(
            side_effect=[
                ValueError("1"),
                ValueError("2"),
                ValueError("3"),
                ValueError("4"),
                "ok",
            ]
        )

        import rag_pipeline.reliability.retry as retry_mod

        original_func = retry_mod.time.sleep
        retry_mod.time.sleep = lambda d: delays.append(d)
        try:
            retry_with_backoff(
                func,
                max_retries=4,
                base_delay=1.0,
                max_delay=3.0,
                exponential=True,
                jitter=False,
            )
        finally:
            retry_mod.time.sleep = original_func

        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 3.0  # capped at max_delay
        assert delays[3] == 3.0  # capped at max_delay

    def test_func_arguments_passed_correctly(self):
        func = MagicMock(return_value="result")
        result = retry_with_backoff(func, "a", "b", x=1, y=2, max_retries=0)
        assert result == "result"
        func.assert_called_once_with("a", "b", x=1, y=2)
