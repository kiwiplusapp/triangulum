"""Core primitives: exact decimal math, domain types, clocks, bus, config."""

from triangulum.core.clock import Clock, LatencyBudget, SimulatedClock, SystemClock
from triangulum.core.config import Config, load_config
from triangulum.core.decimal_math import D, ONE, ZERO, bps, from_bps
from triangulum.core.eventbus import EventBus, Event, Topics
from triangulum.core.types import (
    Asset,
    Fill,
    Leg,
    Opportunity,
    Order,
    OrderType,
    Side,
    Symbol,
    TimeInForce,
)

__all__ = [
    "Asset", "Symbol", "Side", "Order", "OrderType", "TimeInForce", "Fill", "Leg",
    "Opportunity", "D", "ZERO", "ONE", "bps", "from_bps", "Clock", "SystemClock",
    "SimulatedClock", "LatencyBudget", "EventBus", "Event", "Topics", "Config",
    "load_config",
]
