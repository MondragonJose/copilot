"""Pure helper functions shared across packages.

All functions in this module are stateless and have zero I/O.
They belong in ``core/`` per the dependency rule.
"""

import math


def clamp_score(value: float) -> float:
    """Clamp *value* to the closed interval ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, value))


def exponential_backoff(
    attempt: int,
    *,
    base_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = math.inf,
) -> float:
    """Exponential backoff delay for retry logic.

    Formula: ``min(base_delay * multiplier ** (attempt - 1), max_delay)``.

    Parameters
    ----------
    attempt
        1-based attempt counter (``attempt=1`` gives ``base_delay``).
    base_delay
        Delay in seconds for the first attempt (default 1.0).
    multiplier
        Exponential factor (default 2.0).
    max_delay
        Cap in seconds (default unlimited).
    """
    delay: float = base_delay * (multiplier ** (attempt - 1))
    return min(delay, max_delay)
