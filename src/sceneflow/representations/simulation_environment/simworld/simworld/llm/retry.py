"""Retry decorator module for handling LLM API call retries with exponential backoff."""

import functools
import time

import openai

from ..utils.logger import Logger

logger = Logger.get_logger('Retry')


class LLMResponseParsingError(Exception):
    """Raised when LLM response parsing fails."""
    pass


def retry_api_call(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    rate_limit_per_min: int = 20,
):
    """Decorator for retrying LLM API calls with exponential backoff.

    Handles API-related errors like network issues, timeouts, and rate limits.

    Args:
        max_retries: Maximum number of retry attempts before giving up.
        initial_delay: Initial delay between retries in seconds.
        exponential_base: Base for exponential backoff calculation.
        rate_limit_per_min: Maximum number of calls allowed per minute.

    Returns:
        Decorated function that implements API retry logic.
    """
    api_exceptions = (
        openai.APIError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        LLMResponseParsingError,
    )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                try:
                    return func(*args, **kwargs)
                except api_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f'API call failed after {max_retries} retries. Last error: {str(e)}')
                        raise

                    wait_time = delay * (exponential_base ** attempt)
                    logger.warning(
                        f'API attempt {attempt + 1}/{max_retries} failed: {str(e)}. '
                        f'Retrying in {wait_time:.2f} seconds...'
                    )
                    time.sleep(wait_time)
                except Exception as e:
                    logger.error(f'API call failed after {max_retries} retries. Last error: {str(e)}')
                    raise

            raise last_exception
        return wrapper
    return decorator
