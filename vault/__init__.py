"""
Vault — an autonomous macro-synthesis agent that keeps score on itself.

The premise, stated up front because it is the whole design:

    A forecasting system that does not record and resolve its own predictions
    is not a forecasting system. It is a generator of plausible sentences.

So the pipeline runs in one direction and cannot be run backwards:

    SCAN     ingest macro and market series from public sources
    MACRO    classify the regime: growth/inflation quadrant, curve, real rates
    FLOW     cross-asset positioning, momentum, correlation state
    BIAS     Claude synthesizes a FALSIFIABLE thesis -- direction, horizon,
             probability, and an explicit invalidation level
    COMMIT   the thesis is hash-chained into an append-only journal BEFORE the
             outcome is knowable
    RESOLVE  at the horizon, the actual price is fetched and the call is marked
    SCORE    Brier score, calibration curve, skill against the base rate
    GATE     position size is a function of DEMONSTRATED calibration, and is
             exactly zero until the agent beats a coin flip with confidence

The gate is the safety mechanism. Not a warning label -- a number the agent
cannot argue with, computed from its own track record.
"""

from vault.version import __version__

__all__ = ["__version__"]
