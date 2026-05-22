"""Core modules for the fake news diffusion simulation."""

from .constants import (
    ROLE_BOT,
    ROLE_FACT_CHECKER,
    ROLE_INFLUENCER,
    ROLE_REGULAR,
    STATE_BELIEVER,
    STATE_INFORMED,
    STATE_SUSCEPTIBLE,
)

__all__ = [
    "STATE_SUSCEPTIBLE",
    "STATE_BELIEVER",
    "STATE_INFORMED",
    "ROLE_REGULAR",
    "ROLE_INFLUENCER",
    "ROLE_FACT_CHECKER",
    "ROLE_BOT",
]
