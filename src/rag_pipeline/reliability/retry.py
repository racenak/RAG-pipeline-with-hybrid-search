"""Retry with exponential backoff — handle transient failures."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
    **kwargs: Any,
) -> T:
    """Execute function with retry and exponential backoff.

    Args:
        func: Function to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries (seconds).
        max_delay: Maximum delay between retries (seconds).
        exponential: Use exponential backoff (True) or linear (False).
        jitter: Add random jitter to delay.
        retryable_exceptions: Tuple of exceptions to retry on.
        on_retry: Callback function(attempt, exception) called on each retry.

    Returns:
        Result of the function call.

    Raises:
        Last exception if all retries exhausted.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e

            if attempt == max_retries:
                func_name = getattr(func, "__name__", repr(func))
                logger.error(
                    "All %d retries exhausted for %s: %s",
                    max_retries,
                    func_name,
                    e,
                )
                raise

            # Calculate delay
            if exponential:
                delay = min(base_delay * (2**attempt), max_delay)
            else:
                delay = min(base_delay * (attempt + 1), max_delay)

            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)

            func_name = getattr(func, "__name__", repr(func))
            logger.warning(
                "Retry %d/%d for %s after %.1fs: %s",
                attempt + 1,
                max_retries,
                func_name,
                delay,
                e,
            )

            if on_retry:
                on_retry(attempt + 1, e)

            time.sleep(delay)

    raise last_exception  # type: ignore[misc]
